import sys
import os
import uuid
import asyncio
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Add root directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.schemas import EvaluationRequest, TaskResponse, TaskStatus
from src.eval_engine import EvaluationEngine

app = FastAPI(title="VSIG-Bench API")

# Create API router with prefix
api_router = APIRouter(prefix="/embodied_benchmark/api")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount results directory for static access
# Assuming the app is run from the project root
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(results_dir, exist_ok=True)
app.mount("/embodied_benchmark/results", StaticFiles(directory=results_dir), name="results")

# Include API router
app.include_router(api_router)

# In-memory task store
tasks = {}

def run_evaluation_task(task_id: str, config: dict):
    tasks[task_id]["status"] = "processing"
    tasks[task_id]["progress"] = 0.0
    
    def status_callback(msg):
        # Update logs
        tasks[task_id]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        # Naive progress update based on messages (could be better)
        if "Processing [" in msg:
            try:
                # Parse [1/10]
                parts = msg.split("[")[1].split("]")[0].split("/")
                current = int(parts[0])
                total = int(parts[1])
                tasks[task_id]["progress"] = (current / total) * 100
            except:
                pass

    try:
        engine = EvaluationEngine(config, status_callback=status_callback)
        result_path = engine.run()
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100.0
        tasks[task_id]["result_path"] = result_path
        tasks[task_id]["message"] = "Evaluation completed successfully."
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = str(e)

@api_router.post("/evaluate", response_model=TaskResponse)
async def submit_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    
    # Create output dir for this task
    output_dir = os.path.join("results", "web_runs", task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    config = request.model_dump()
    config["output_dir"] = output_dir
    # Ensure test_mode is explicitly in config
    config["test_mode"] = request.test_mode
    
    tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Task queued.",
        "logs": [],
        "result_path": None,
        "config": config
    }
    
    background_tasks.add_task(run_evaluation_task, task_id, config)
    
    return TaskResponse(
        task_id=task_id,
        status="pending",
        message="Evaluation task started."
    )

@api_router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatus(**tasks[task_id])

@api_router.get("/results/{task_id}")
async def get_task_results(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task["status"] != "completed":
         raise HTTPException(status_code=400, detail="Task not completed")
         
    result_path = task["result_path"]
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")
        
    import json
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6006)

