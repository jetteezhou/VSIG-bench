from pydantic import BaseModel, Field
from typing import Optional, List, Union

class EvaluationRequest(BaseModel):
    # API Configuration
    api_provider: str = Field(..., description="API Provider: 'openai' or 'gemini'")
    api_base_url: Optional[str] = Field(None, description="API Base URL (for OpenAI compatible)")
    api_key: str = Field(..., description="API Key")
    model_name: str = Field(..., description="Model Name (e.g., gpt-4o, gemini-1.5-pro)")
    
    # Input Configuration
    input_mode: str = Field("video", description="'video' or 'frames'")
    num_frames: int = Field(8, description="Number of frames to extract if input_mode is 'frames'")
    
    # Optional Parameters
    system_prompt: Optional[str] = Field(None, description="Custom System Prompt")
    temperature: float = Field(0.2, description="Sampling temperature")
    max_tokens: int = Field(1000, description="Max tokens for generation")
    
    # Evaluation Config
    data_root_dir: str = Field("data", description="Root directory of the dataset")

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    status: str # pending, processing, completed, failed
    progress: float
    message: Optional[str] = None
    result_path: Optional[str] = None
    logs: List[str] = []

