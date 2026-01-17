import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.prompts.vsig_prompts import VSIGPrompts
from src.models.base_vlm import OpenAIVLM, GeminiVLM
from src.utils.video_processor import VideoProcessor
from src.eval.metrics import Evaluator

# Configure logger
logger = logging.getLogger("VSIG_Engine")
logger.setLevel(logging.INFO)

import threading

# Global lock for the static Evaluator class to prevent race conditions on _eval_model
evaluator_lock = threading.Lock()

class EvaluationEngine:
    def __init__(self, config, status_callback=None):
        self.config = config
        self.status_callback = status_callback
        self.model = None
        self.eval_model = None

    def log(self, message, level="info"):
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        
        if self.status_callback:
            self.status_callback(message)

    def init_models(self):
        # Initialize Inference Model
        provider = self.config.get("model_provider")
        model_name = self.config.get("model_name")
        api_key = self.config.get("api_key")
        base_url = self.config.get("api_base_url")
        
        input_mode = self.config.get("input_mode", "frames")
        use_video_input = (input_mode == "video")

        self.log(f"Initializing Model: {provider}/{model_name} (Video Input: {use_video_input})")

        if provider == "openai":
            self.model = OpenAIVLM(
                api_key=api_key, 
                base_url=base_url, 
                model_name=model_name,
                accepts_video_files=use_video_input
            )
        elif provider == "gemini":
            self.model = GeminiVLM(
                api_key=api_key,
                model_name=model_name
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Initialize Eval Model (Use same as inference for now to simplify)
        self.eval_model = self.model
        # Note: We do NOT set Evaluator.set_eval_model here anymore to avoid race conditions.
        # It will be set inside the locked evaluation block.

    def process_single_sample(self, item, video_dir, output_dir, idx, total):
        video_id = item["video_name"]
        self.log(f"Processing [{idx+1}/{total}]: {video_id}")

        video_path = os.path.join(video_dir, video_id)
        
        # ASR handling
        asr_result = item.get("asr_result")
        transcript = asr_result["text"] if (asr_result and "text" in asr_result) else item.get("task_template")
        if item.get("task_template") == "指令1":
            transcript = "用户没有说话，只是做出了指向性动作。"

        # Extract Input
        frame_paths = []
        last_frame_path = None
        
        use_video_input = (self.config.get("input_mode") == "video")
        num_frames = self.config.get("num_frames", 8)

        try:
            if use_video_input and hasattr(self.model, 'generate_from_video') and getattr(self.model, 'accepts_video_files', False):
                 _, last_frame_path = VideoProcessor.extract_frame(video_path, timestamp_sec=None)
            else:
                 frame_paths, last_frame_path = VideoProcessor.extract_frames(
                    video_path, num_frames=num_frames, end_timestamp_sec=item.get("timestamp"))
        except Exception as e:
            self.log(f"Error processing video {video_id}: {e}", "error")
            return None, None

        # Build Prompts
        system_prompt = self.config.get("system_prompt")
        if not system_prompt:
            system_prompt = VSIGPrompts.get_system_prompt(task_template=item.get("task_template"))
        
        user_prompt = VSIGPrompts.get_user_prompt(transcript, asr_result=asr_result)

        # Inference
        try:
            if use_video_input and hasattr(self.model, 'generate_from_video') and getattr(self.model, 'accepts_video_files', False):
                result = self.model.generate_from_video(video_path, user_prompt, system_prompt=system_prompt)
            else:
                result = self.model.generate(frame_paths, user_prompt, system_prompt=system_prompt)
        except Exception as e:
            self.log(f"Inference error {video_id}: {e}", "error")
            return None, None

        if not result or not isinstance(result, (dict, list)):
            return None, None
            
        if isinstance(result, list):
            result = result[0] if len(result) > 0 and isinstance(result[0], dict) else {}

        result["video_name"] = video_id
        
        # Visualize (Optional, skipping for speed/simplicity in web mode, or can enable)
        # vis_path = os.path.join(output_dir, f"vis_{video_id}.jpg")
        # VideoProcessor.visualize_points(last_frame_path, result, vis_path, gt_json=item)

        return result, item

    def run(self):
        try:
            self.init_models()
            
            data_root = self.config.get("data_root_dir", "data")
            output_dir = self.config.get("output_dir", "results/web_run")
            
            # Simple scan logic (simplified from main.py)
            dataset_dirs = []
            if os.path.basename(data_root) in ["data", "data_new"] or data_root.endswith("/data"):
                for d in os.listdir(data_root):
                    dp = os.path.join(data_root, d)
                    if os.path.isdir(dp): dataset_dirs.append((d, dp))
            else:
                dataset_dirs.append((os.path.basename(data_root), data_root))

            all_predictions = []
            all_ground_truths = []

            for dataset_name, dataset_path in dataset_dirs:
                self.log(f"Scanning dataset: {dataset_name}")
                if not os.path.exists(dataset_path): continue

                for item in os.listdir(dataset_path):
                    item_path = os.path.join(dataset_path, item)
                    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "annotations.json")):
                        if item.startswith("指令"):
                            self.process_directory(dataset_name, dataset_path, item_path, output_dir, all_predictions, all_ground_truths)

            # Evaluate
            if all_predictions:
                self.log("Starting Evaluation (Acquiring Lock)...")
                with evaluator_lock:
                    self.log("Lock Acquired. Configuring Evaluator...")
                    Evaluator.set_eval_model(self.eval_model)
                    
                    self.log(f"Calculating metrics for {len(all_predictions)} samples...")
                    metrics = Evaluator.evaluate_batch(all_predictions, all_ground_truths, num_workers=4)
                
                metrics_path = os.path.join(output_dir, "metrics.json")
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                
                self.log("Evaluation Complete.")
                return metrics_path
            else:
                self.log("No predictions generated.", "warning")
                return None

        except Exception as e:
            self.log(f"Critical Error: {e}", "error")
            import traceback
            self.log(traceback.format_exc(), "error")
            raise e

    def process_directory(self, dataset_name, dataset_path, instruction_path, output_dir, all_preds, all_gts):
        instruction_name = os.path.basename(instruction_path)
        self.log(f"Processing {dataset_name} / {instruction_name}")
        
        meta_file = os.path.join(instruction_path, "annotations.json")
        with open(meta_file, 'r', encoding='utf-8') as f:
            all_dataset = json.load(f)
            
        video_dir = instruction_path
        video_files = set(os.listdir(video_dir))
        
        dataset = []
        for d in all_dataset:
            if d["video_name"] in video_files:
                d["_video_dir"] = video_dir
                dataset.append(d)
                
        # Run inference
        results_dir = os.path.join(output_dir, instruction_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # Parallel Execution for Inference
        # Use a reasonable number of workers (e.g., 4-8) to avoid rate limits or OOM
        num_workers = self.config.get("num_workers", 4)
        
        self.log(f"Starting inference with {num_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_item = {
                executor.submit(self.process_single_sample, item, video_dir, results_dir, idx, len(dataset)): item
                for idx, item in enumerate(dataset)
            }
            
            for future in as_completed(future_to_item):
                try:
                    pred, gt = future.result()
                    if pred and gt:
                        all_preds.append(pred)
                        all_gts.append(gt)
                except Exception as e:
                    self.log(f"Sample processing failed: {e}", "error")


