import threading
import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.prompts.vsig_prompts import VSIGPrompts
from src.models.base_vlm import OpenAIVLM, GeminiVLM
from src.utils.video_processor import VideoProcessor
from src.eval.metrics import Evaluator
from src.data_loader import DataLoader
from src.gt_formatter import GTFormatter
from src.utils.logger import setup_logger

# Configure logger (will be configured with file output in __init__)
logger = logging.getLogger("VSIG_Engine")
logger.setLevel(logging.INFO)


# Global lock for the static Evaluator class to prevent race conditions on _eval_model
evaluator_lock = threading.Lock()


class EvaluationEngine:
    def __init__(self, config, status_callback=None):
        self.config = config
        self.status_callback = status_callback
        self.model = None
        self.eval_model = None
        
        # 配置logger，将日志保存到output_dir
        output_dir = config.get("output_dir", "results/web_run")
        log_dir = os.path.join(output_dir, "logs")
        # 使用全局logger，但添加文件handler
        global logger
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            # 如果还没有文件handler，添加一个
            os.makedirs(log_dir, exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"eval_{timestamp}.log")
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            # 确保有控制台handler
            if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            logger.info(f"日志文件: {log_file}")

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

        self.log(
            f"Initializing Model: {provider}/{model_name} (Video Input: {use_video_input})")

        coord_order = self.config.get("coord_order", "xy")

        if provider == "openai":
            self.model = OpenAIVLM(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                accepts_video_files=use_video_input,
                coord_order=coord_order
            )
        elif provider == "gemini":
            self.model = GeminiVLM(
                api_key=api_key,
                model_name=model_name,
                coord_order=coord_order
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Initialize Eval Model (Use separate eval model if configured, otherwise use inference model)
        eval_model_provider = self.config.get("eval_model_provider")
        eval_model_name = self.config.get("eval_model_name")

        if eval_model_provider and eval_model_name:
            # Use separate evaluation model
            self.log(
                f"Initializing Evaluation Model: {eval_model_provider}/{eval_model_name}")

            eval_api_key = self.config.get("eval_api_key") or api_key
            eval_base_url = self.config.get("eval_api_base_url") or base_url

            if eval_model_provider == "openai":
                if not eval_api_key:
                    self.log("未找到评估用 OpenAI API Key，将使用推理模型进行评估", "warning")
                    self.eval_model = self.model
                else:
                    self.eval_model = OpenAIVLM(
                        api_key=eval_api_key,
                        base_url=eval_base_url,
                        model_name=eval_model_name,
                        accepts_video_files=False,  # Eval model typically doesn't need video input
                        coord_order=coord_order
                    )
                    self.log(f"评估模型已设置为: {eval_model_name} (OpenAI)")

            elif eval_model_provider == "gemini":
                if not eval_api_key:
                    self.log("未找到评估用 Gemini API Key，将使用推理模型进行评估", "warning")
                    self.eval_model = self.model
                else:
                    self.eval_model = GeminiVLM(
                        api_key=eval_api_key,
                        model_name=eval_model_name,
                        coord_order=coord_order
                    )
                    self.log(f"评估模型已设置为: {eval_model_name} (Gemini)")

            else:
                self.log(
                    f"不支持的评估模型提供商: {eval_model_provider}，将使用推理模型进行评估", "warning")
                self.eval_model = self.model
        else:
            # No separate eval model configured, use inference model
            self.log("未配置单独的评估模型，将使用推理模型进行评估")
            self.eval_model = self.model

        # Note: We do NOT set Evaluator.set_eval_model here anymore to avoid race conditions.
        # It will be set inside the locked evaluation block.

    def process_single_sample(self, formatted_gt, options_text, output_dir, idx, total):
        """
        处理单个视频样本

        Args:
            formatted_gt: 已格式化的GT数据
            options_text: 选项定义文本
            output_dir: 结果保存目录
            idx: 当前样本索引
            total: 总样本数
        """
        video_id = formatted_gt["video_name"]
        video_dir = formatted_gt["_video_dir"]
        self.log(f"Processing [{idx+1}/{total}]: {video_id}")

        video_path = os.path.join(video_dir, video_id)

        # ASR handling
        asr_result = formatted_gt.get("asr_result")
        transcript = asr_result["text"] if (
            asr_result and isinstance(asr_result, dict) and "text" in asr_result) else formatted_gt.get("task_template")

        # 指令1 ASR text 有时候是 null 或 ""，统一处理
        if formatted_gt.get("task_template") == "指令1" or not transcript:
            transcript = "用户没有说话，只是做出了指向性动作。"

        # Extract Input
        frame_paths = []
        last_frame_path = None

        use_video_input = (self.config.get("input_mode") == "video")
        num_frames = self.config.get("num_frames", 15)

        try:
            if use_video_input and hasattr(self.model, 'generate_from_video') and getattr(self.model, 'accepts_video_files', False):
                _, last_frame_path = VideoProcessor.extract_frame(
                    video_path, timestamp_sec=None)
            else:
                frame_paths, last_frame_path = VideoProcessor.extract_frames(
                    video_path, num_frames=num_frames, end_timestamp_sec=formatted_gt.get("timestamp"))
        except Exception as e:
            self.log(f"Error processing video {video_id}: {e}", "error")
            return None, None

        # Build Prompts
        system_prompt = self.config.get("system_prompt")
        if not system_prompt:
            coord_order = self.config.get("coord_order", "xy")
            system_prompt = VSIGPrompts.get_system_prompt(
                task_template=formatted_gt.get("task_template"),
                coord_order=coord_order,
                options_text=options_text
            )

        user_prompt = VSIGPrompts.get_user_prompt(
            transcript, asr_result=asr_result)

        # Inference
        try:
            if use_video_input and hasattr(self.model, 'generate_from_video') and getattr(self.model, 'accepts_video_files', False):
                result = self.model.generate_from_video(
                    video_path, user_prompt, system_prompt=system_prompt)
            else:
                result = self.model.generate(
                    frame_paths, user_prompt, system_prompt=system_prompt)
        except Exception as e:
            self.log(f"Inference error {video_id}: {e}", "error")
            return None, None

        if not result or not isinstance(result, (dict, list)):
            return None, None

        if isinstance(result, list):
            result = result[0] if len(result) > 0 and isinstance(
                result[0], dict) else {}

        result["video_name"] = video_id

        # 注意：坐标转换已在模型类的 generate/generate_from_video 方法中完成
        # result["point_list"] 中的所有 points 都已经是 [x, y] 格式

        # Visualize (Enabled in test mode OR if running a web task)
        is_web_run = "web_runs" in output_dir
        if self.config.get("test_mode", False) or is_web_run:
            vis_filename = f"vis_{video_id}.jpg"
            vis_path = os.path.join(output_dir, vis_filename)
            try:
                # 使用已格式化的GT数据
                processed_gt = formatted_gt.get("_processed_gt", {})
                gt_items = processed_gt.get("items", [])

                VideoProcessor.visualize_points(
                    last_frame_path, result, vis_path, gt_json=formatted_gt, gt_items=gt_items)

                # Add relative path for frontend access (relative to the results root directory)
                # 尝试根据 Config.OUTPUT_DIR 或常见结果目录名称提取相对路径
                results_keyword = "results"
                from config import Config
                if hasattr(Config, 'OUTPUT_DIR') and Config.OUTPUT_DIR:
                    results_keyword = Config.OUTPUT_DIR.split('/')[0]
                
                if results_keyword in vis_path:
                    rel_parts = vis_path.split(f"{results_keyword}/")
                    if len(rel_parts) > 1:
                        result["visualization_rel_path"] = rel_parts[-1]
                    else:
                        result["visualization_rel_path"] = vis_path
                else:
                    # 如果找不到关键字，尝试查找是否有带 -2, -3 等后缀的 results
                    import re
                    match = re.search(r'results[^/]*', vis_path)
                    if match:
                        keyword = match.group(0)
                        rel_parts = vis_path.split(f"{keyword}/")
                        if len(rel_parts) > 1:
                            result["visualization_rel_path"] = rel_parts[-1]
                        else:
                            result["visualization_rel_path"] = vis_path
                    else:
                        result["visualization_rel_path"] = vis_path
            except Exception as e:
                self.log(
                    f"Visualization failed for {video_id}: {e}", "warning")

        return result, formatted_gt

    def run(self):
        try:
            self.init_models()

            data_root = self.config.get("data_root_dir", Config.DATA_ROOT_DIR)
            
            # 优先从 Config.OUTPUT_DIR 获取根目录
            results_root = "results"
            if Config.OUTPUT_DIR:
                if "/" in Config.OUTPUT_DIR or "\\" in Config.OUTPUT_DIR:
                    results_root = os.path.dirname(Config.OUTPUT_DIR)
                else:
                    results_root = Config.OUTPUT_DIR
            
            output_dir = self.config.get("output_dir", os.path.join(results_root, "web_run"))

            # Simple scan logic (simplified from main.py)
            dataset_dirs = []
            if os.path.basename(data_root).startswith("data"):
                for d in os.listdir(data_root):
                    dp = os.path.join(data_root, d)
                    if os.path.isdir(dp):
                        dataset_dirs.append((d, dp))
            else:
                dataset_dirs.append((os.path.basename(data_root), data_root))

            all_predictions = []
            all_ground_truths = []
            all_tasks = []

            for dataset_name, dataset_path in dataset_dirs:
                self.log(f"Scanning dataset: {dataset_name}")
                if not os.path.exists(dataset_path):
                    continue

                for instruction_dir in sorted(os.listdir(dataset_path)):
                    instruction_path = os.path.join(
                        dataset_path, instruction_dir)
                    if os.path.isdir(instruction_path) and os.path.exists(os.path.join(instruction_path, "annotations.json")):
                        if instruction_dir.startswith("指令"):
                            tasks_in_dir = self.collect_tasks(
                                dataset_name, dataset_path, instruction_path, output_dir)
                            all_tasks.extend(tasks_in_dir)

            if not all_tasks:
                self.log("No tasks found.", "warning")
                return None

            # Parallel Execution for Inference
            num_workers = self.config.get("num_workers", 20)
            self.log(
                f"Starting inference for {len(all_tasks)} total samples with {num_workers} workers...")

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_task = {
                    executor.submit(
                        self.process_single_sample,
                        task["formatted_gt"],
                        task["options_text"],
                        task["results_dir"],
                        idx,
                        len(all_tasks)
                    ): task
                    for idx, task in enumerate(all_tasks)
                }

                for future in as_completed(future_to_task):
                    try:
                        pred, gt = future.result()
                        if pred and gt:
                            all_predictions.append(pred)
                            all_ground_truths.append(gt)
                    except Exception as e:
                        task = future_to_task[future]
                        self.log(f"Sample processing failed: {e}", "error")

            # Evaluate
            if all_predictions:
                self.log("Starting Evaluation...")
                
                self.log(
                    f"Calculating metrics for {len(all_predictions)} samples...")
                metrics = Evaluator.evaluate_batch(
                    all_predictions, all_ground_truths, num_workers=10)

                # 移除冗余的第二次并行评估，直接使用 metrics 中的 detailed_results
                is_web_run = "web_runs" in output_dir
                if not (self.config.get("test_mode", False) or is_web_run):
                    # 如果不是测试模式或 Web 运行，为了节省空间可以删除详情
                    if "detailed_results" in metrics:
                        del metrics["detailed_results"]
                else:
                    if is_web_run:
                        self.log(
                            "Web Run: Using pre-calculated detailed evaluation results.")
                    else:
                        self.log(
                            "Test Mode: Using pre-calculated detailed evaluation results.")

                # Inject model metadata into metrics
                metrics["model_name"] = self.config.get(
                    "model_name", "Unknown")
                metrics["model_provider"] = self.config.get(
                    "model_provider", "Unknown")

                metrics_path = os.path.join(output_dir, "metrics.json")
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)

                # Save metrics_summary.json for leaderboard compatibility
                summary_data = {
                    "model_name": self.config.get("model_name", "Unknown"),
                    "model_provider": self.config.get("model_provider", "Unknown"),
                    "overall": metrics
                }

                summary_path = os.path.join(output_dir, "metrics_summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary_data, f, indent=2, ensure_ascii=False)

                self.log("Evaluation Complete.")
                return metrics_path
            else:
                error_msg = "No predictions were generated. Please check your API key, model name, and network connection in the Execution Log."
                self.log(error_msg, "error")
                raise ValueError(error_msg)

        except Exception as e:
            self.log(f"Critical Error: {e}", "error")
            import traceback
            self.log(traceback.format_exc(), "error")
            raise e

    def collect_tasks(self, dataset_name, dataset_path, instruction_path, output_dir):
        """
        收集任务：统一使用 DataLoader 和 GTFormatter

        Returns:
            List of tasks, each containing:
            - formatted_gt: 已格式化的GT数据
            - options_text: 选项定义文本
            - results_dir: 结果保存目录
        """
        instruction_name = os.path.basename(instruction_path)
        meta_file = os.path.join(instruction_path, "annotations.json")
        video_dir = instruction_path

        # 1. 使用 DataLoader 读取数据
        # 现在 prepare_dataset 内部会自动寻找 eval_gt.json
        dataset, options_text, video_eval_data = DataLoader.prepare_dataset(
            video_dir=video_dir,
            annotation_path=meta_file
        )

        # Test mode: only take the first sample for each instruction
        if self.config.get("test_mode", False) and len(dataset) > 0:
            dataset = dataset[:1]

        # 2. 使用 GTFormatter 格式化GT数据
        formatted_gt_list = GTFormatter.format_batch_gt_for_evaluation(
            dataset, video_dir, video_eval_data
        )

        # 如果 video_eval_data 不为空，说明会过滤掉不在评估列表中的视频
        if video_eval_data and len(video_eval_data) > 0:
            skipped_count = len(dataset) - len(formatted_gt_list)
            if skipped_count > 0:
                self.log(
                    f"已过滤掉 {skipped_count} 个不符合标准的视频（不在 eval_gt.json 中）")

        results_dir = os.path.join(output_dir, instruction_name)
        os.makedirs(results_dir, exist_ok=True)

        return [
            {
                "formatted_gt": formatted_gt,
                "options_text": options_text,
                "results_dir": results_dir
            }
            for formatted_gt in formatted_gt_list
        ]
