# main.py
import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config
from src.prompts.vsig_prompts import VSIGPrompts
from src.models.base_vlm import OpenAIVLM, GeminiVLM
from src.utils.video_processor import VideoProcessor
from src.eval.metrics import Evaluator
from src.utils.logger import setup_logger


def process_single_sample(item, video_dir, output_dir, model, logger, idx, total):
    """
    处理单个视频样本

    Args:
        item: 数据集项（包含 video_name, task_template 等信息）
        video_dir: 视频存放目录
        output_dir: 结果保存目录
        model: 模型实例
        logger: 日志记录器
        idx: 当前样本索引（从0开始）
        total: 总样本数

    Returns:
        tuple: (prediction, ground_truth) 或 (None, None) 如果处理失败
    """
    video_id = item["video_name"]
    logger.info(f"[{idx+1}/{total}] 处理样本: {video_id}")

    video_path = os.path.join(video_dir, video_id)

    # 获取 ASR 文本，如果没有则使用任务模板描述
    asr_result = item.get("asr_result")
    transcript = asr_result["text"] if (
        asr_result and isinstance(asr_result, dict) and "text" in asr_result) else item.get("task_template")

    if item.get("task_template") == "指令1":
        transcript = "用户没有说话，只是做出了指向性动作。"

    # 4.1 提取帧 / 准备输入
    last_frame_path = None
    frame_paths = []

    try:
        if Config.USE_VIDEO_INPUT and hasattr(model, 'generate_from_video') and getattr(model, 'accepts_video_files', False):
            # 如果配置了视频输入且模型支持
            logger.info(f"使用直接视频输入模式: {video_path}")
            # 提取最后一帧用于可视化 (尝试提取，失败则忽略)
            try:
                _, last_frame_path = VideoProcessor.extract_frame(
                    video_path, timestamp_sec=None)
            except Exception as vid_e:
                logger.warning(f"无法提取可视化帧 (非致命错误): {vid_e}")
                last_frame_path = None
        else:
            # 默认或不支持视频输入时，使用抽帧模式
            if Config.USE_VIDEO_INPUT:
                logger.warning(f"配置了视频输入但模型 {Config.MODEL_NAME} 不支持，回退到抽帧模式")

            # 提取多帧，默认8帧
            frame_paths, last_frame_path = VideoProcessor.extract_frames(
                video_path, num_frames=Config.NUM_FRAMES, end_timestamp_sec=item.get("timestamp"))
    except Exception as e:
        logger.error(f"处理视频 {video_id} 失败: {e}")
        return None, None

    # 4.2 构建 Prompt
    system_prompt = VSIGPrompts.get_system_prompt(
        task_template=item.get("task_template"))
    user_prompt = VSIGPrompts.get_user_prompt(
        transcript, asr_result=asr_result)

    # 4.3 模型推理
    try:
        if Config.USE_VIDEO_INPUT and hasattr(model, 'generate_from_video') and getattr(model, 'accepts_video_files', False):
            result = model.generate_from_video(
                video_path, user_prompt, system_prompt=system_prompt)
        else:
            # 传入所有帧的路径列表
            result = model.generate(
                frame_paths, user_prompt, system_prompt=system_prompt)
    except Exception as e:
        logger.error(f"样本 {video_id} 推理出错: {e}")
        return None, None

    if not result:
        logger.warning(f"样本 {video_id} 推理无结果，跳过")
        return None, None

    # 4.4 处理 result 格式（可能是字典或列表）
    # 处理 result 可能是字典或列表的情况
    if isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], dict):
            # 如果返回的是列表且第一个元素是字典，使用第一个元素
            result = result[0]
            logger.warning(f"样本 {video_id} 返回结果为列表格式，已提取第一个元素")
        else:
            logger.warning(f"样本 {video_id} 返回结果为列表但格式不正确，跳过")
            return None, None

    if not isinstance(result, dict):
        logger.warning(f"样本 {video_id} 返回结果格式不正确: {type(result)}，跳过")
        return None, None

    # 将视频名称添加到预测结果中
    result["video_name"] = video_id

    # Post-process: Swap coordinates from [x, y] (Prompt/Model) to [y, x] (Eval/Vis/GT)
    # The prompt asks for [x, y] (Width, Height), but the evaluator and visualizer expect [y, x] (Height, Width).
    if "point_list" in result:
        for pred_item in result["point_list"]:
            if "point" in pred_item and isinstance(pred_item["point"], list):
                pt = pred_item["point"]
                # Case 1: Single point [x, y]
                if len(pt) == 2 and isinstance(pt[0], (int, float)):
                    pred_item["point"] = [pt[1], pt[0]]
                # Case 2: Multiple points [[x1, y1], [x2, y2]]
                elif len(pt) > 0 and isinstance(pt[0], list) and len(pt[0]) == 2:
                    pred_item["point"] = [[p[1], p[0]] for p in pt]

    vis_path = os.path.join(output_dir, f"vis_{video_id}.jpg")
    # 可视化依然使用最后一帧（最接近指令结束时刻）
    # 传入 processed_gt (gt_items) 进行对比可视化
    try:
        # Process GT items to match evaluation logic (skip objects, merge names etc.)
        processed_gt = Evaluator.process_gt_by_template(item)
        gt_items = processed_gt.get("items", [])
        VideoProcessor.visualize_points(
            last_frame_path, result, vis_path, gt_json=item, gt_items=gt_items)
    except Exception as e:
        logger.error(f"样本 {video_id} 可视化失败: {e}")

    explicit_cmd = result.get('explicit_command', 'None')
    logger.info(f"样本 {video_id} 完成。指令: {explicit_cmd}")

    return result, item


def process_single_directory(video_dir, meta_file, output_dir, model, logger):
    """
    处理单个指令文件夹

    Args:
        video_dir: 视频存放目录
        meta_file: 标注文件路径
        output_dir: 结果保存目录
        model: 模型实例
        logger: 日志记录器

    Returns:
        predictions: 预测结果列表
        ground_truths: 真实标签列表
    """
    # 3. 加载数据
    if not os.path.exists(meta_file):
        logger.warning(f"标注文件不存在: {meta_file}，跳过该目录")
        return [], []

    with open(meta_file, 'r', encoding='utf-8') as f:
        all_dataset = json.load(f)

    # 提取视频文件名（支持 mp4 和 MOV 格式）
    video_extensions = ['.mp4', '.MOV', '.mov']
    video_file_names = [x for x in os.listdir(video_dir) if any(
        x.endswith(ext) for ext in video_extensions)]
    dataset = []
    # 提取视频文件名对应的标注
    for dataset_item in all_dataset:
        video_name = dataset_item["video_name"]
        if video_name in video_file_names:
            # 添加临时字段用于评估时定位视频分辨率
            dataset_item["_video_dir"] = video_dir
            dataset.append(dataset_item)

    logger.info(f"成功加载数据集，共 {len(dataset)} 条样本")

    predictions = []
    ground_truths = []

    os.makedirs(output_dir, exist_ok=True)

    # 4. 推理循环（支持多线程并行）
    logger.info(f"开始推理循环... (使用 {Config.NUM_WORKERS} 个线程并行处理)")

    if Config.NUM_WORKERS > 1:
        # 多线程并行处理
        with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(process_single_sample, item, video_dir, output_dir, model, logger, idx, len(dataset)): item
                for idx, item in enumerate(dataset)
            }

            # 收集结果
            for future in as_completed(future_to_item):
                try:
                    prediction, ground_truth = future.result()
                    if prediction is not None and ground_truth is not None:
                        predictions.append(prediction)
                        ground_truths.append(ground_truth)
                except Exception as e:
                    item = future_to_item[future]
                    logger.error(
                        f"处理样本 {item.get('video_name', 'unknown')} 时发生异常: {e}")
    else:
        # 单线程串行处理（保持原有逻辑）
        for idx, item in enumerate(dataset):
            prediction, ground_truth = process_single_sample(
                item, video_dir, output_dir, model, logger, idx, len(dataset))
            if prediction is not None and ground_truth is not None:
                predictions.append(prediction)
                ground_truths.append(ground_truth)

    logger.info(f"推理完成，成功处理 {len(predictions)}/{len(dataset)} 个样本")

    return predictions, ground_truths


def main():
    # 0. 初始化 Logger
    output_dir = Config.OUTPUT_DIR if Config.SAVE_LOG else None
    logger = setup_logger(output_dir=output_dir)
    logger.info("Visual-Speech Intent Grounding (VSIG) 任务启动")
    logger.info("加载配置...")

    # 1. 初始化模型
    model_provider = Config.MODEL_PROVIDER
    model_name = Config.MODEL_NAME

    if model_provider == "openai":
        api_key = Config.OPENAI_API_KEY
        base_url = Config.OPENAI_BASE_URL

        if not api_key:
            logger.error(
                "未找到 OpenAI API Key。请在 config.py 中配置 OPENAI_API_KEY 或设置环境变量。")
            sys.exit(1)

        logger.info(f"正在初始化 OpenAIVLM (Model: {model_name})")
        # 如果配置了视频输入模式，则设置 accepts_video_files=True
        accepts_video = getattr(Config, 'USE_VIDEO_INPUT', False)
        model = OpenAIVLM(api_key=api_key, base_url=base_url,
                          model_name=model_name, accepts_video_files=accepts_video)

    elif model_provider == "gemini":
        api_key = Config.GEMINI_API_KEY

        if not api_key:
            logger.error(
                "未找到 Gemini API Key。请在 config.py 中配置 GEMINI_API_KEY 或设置环境变量。")
            sys.exit(1)

        logger.info(f"正在初始化 GeminiVLM (Model: {model_name})")
        model = GeminiVLM(api_key=api_key, model_name=model_name)

    else:
        logger.error(f"不支持的模型提供商: {model_provider}")
        sys.exit(1)

    # 初始化评估模型（如果配置了单独的评估模型，则使用评估模型；否则使用推理模型）
    eval_model = None
    eval_model_provider = Config.EVAL_MODEL_PROVIDER
    eval_model_name = Config.EVAL_MODEL_NAME

    if eval_model_provider and eval_model_name:
        # 使用单独的评估模型配置
        logger.info(f"正在初始化评估模型 ({eval_model_provider}: {eval_model_name})...")

        if eval_model_provider == "openai":
            eval_api_key = Config.EVAL_OPENAI_API_KEY or Config.OPENAI_API_KEY
            eval_base_url = Config.EVAL_OPENAI_BASE_URL or Config.OPENAI_BASE_URL

            if not eval_api_key:
                logger.warning("未找到评估用 OpenAI API Key，将使用推理模型进行评估")
                eval_model = model
            else:
                eval_model = OpenAIVLM(api_key=eval_api_key, base_url=eval_base_url,
                                       model_name=eval_model_name)
                logger.info(f"评估模型已设置为: {eval_model_name} (OpenAI)")

        elif eval_model_provider == "gemini":
            eval_api_key = Config.EVAL_GEMINI_API_KEY or Config.GEMINI_API_KEY

            if not eval_api_key:
                logger.warning("未找到评估用 Gemini API Key，将使用推理模型进行评估")
                eval_model = model
            else:
                eval_model = GeminiVLM(
                    api_key=eval_api_key, model_name=eval_model_name)
                logger.info(f"评估模型已设置为: {eval_model_name} (Gemini)")

        else:
            logger.warning(f"不支持的评估模型提供商: {eval_model_provider}，将使用推理模型进行评估")
            eval_model = model
    else:
        # 未配置单独的评估模型，使用推理模型进行评估
        logger.info("未配置单独的评估模型，将使用推理模型进行评估")
        eval_model = model

    # 为评估器设置模型助手
    Evaluator.set_eval_model(eval_model)

    # 2. 准备数据
    data_root_dir = Config.DATA_ROOT_DIR
    base_output_dir = Config.OUTPUT_DIR

    if not os.path.exists(data_root_dir):
        logger.critical(f"数据根目录不存在: {data_root_dir}")
        sys.exit(1)

    # 扫描所有数据集和指令文件夹
    # 数据结构: {指令名: [(数据集路径, 指令文件夹路径), ...]}
    instruction_dirs = {}
    dataset_dirs = []

    # 如果 data_root_dir 是 "data" 或 "data_new"，扫描其下所有子目录作为数据集
    if os.path.basename(data_root_dir) in ["data", "data_new"] or data_root_dir.endswith("/data") or data_root_dir.endswith("/data_new"):
        # 扫描 data/ 下的所有数据集目录
        for dataset_name in os.listdir(data_root_dir):
            dataset_path = os.path.join(data_root_dir, dataset_name)
            if os.path.isdir(dataset_path):
                dataset_dirs.append((dataset_name, dataset_path))
    else:
        # 单个数据集目录
        dataset_dirs.append((os.path.basename(data_root_dir), data_root_dir))

    logger.info(
        f"找到 {len(dataset_dirs)} 个数据集目录: {[d[0] for d in dataset_dirs]}")

    # 扫描所有数据集中的指令文件夹
    for dataset_name, dataset_path in dataset_dirs:
        logger.info(f"扫描数据集: {dataset_name} (路径: {dataset_path})")
        if not os.path.exists(dataset_path):
            logger.warning(f"数据集路径不存在: {dataset_path}，跳过")
            continue

        items_found = []
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "annotations.json")):
                # 检查是否是指令文件夹（指令1-6）
                if item.startswith("指令") and item in ["指令1", "指令2", "指令3", "指令4", "指令5", "指令6"]:
                    if item not in instruction_dirs:
                        instruction_dirs[item] = []
                    instruction_dirs[item].append(
                        (dataset_name, dataset_path, item_path))
                    items_found.append(item)

        logger.info(
            f"数据集 {dataset_name} 中找到 {len(items_found)} 个指令文件夹: {items_found}")

    if not instruction_dirs:
        logger.warning(f"未找到任何指令文件夹（指令1-指令6）")
        sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info(f"扫描结果汇总:")
    logger.info(f"{'='*60}")
    logger.info(f"找到以下指令类型: {sorted(instruction_dirs.keys())}")
    for inst_name, dirs_list in instruction_dirs.items():
        dataset_names = [d[0] for d in dirs_list]
        logger.info(f"  {inst_name}: {len(dirs_list)} 个数据集 - {dataset_names}")

    # 验证是否所有数据集都被扫描到
    all_scanned_datasets = set()
    for dirs_list in instruction_dirs.values():
        for dataset_name, _, _ in dirs_list:
            all_scanned_datasets.add(dataset_name)

    logger.info(f"\n扫描到的数据集列表: {sorted(all_scanned_datasets)}")
    logger.info(f"期望的数据集数量: {len(dataset_dirs)}")
    if len(all_scanned_datasets) < len(dataset_dirs):
        missing_datasets = set(d[0]
                               for d in dataset_dirs) - all_scanned_datasets
        logger.warning(f"⚠️  警告：以下数据集没有被扫描到: {missing_datasets}")
    logger.info(f"{'='*60}\n")

    # 存储所有目录的预测结果和真实标签（按指令分组）
    all_predictions_by_instruction = {}  # {指令名: [predictions]}
    all_ground_truths_by_instruction = {}  # {指令名: [ground_truths]}
    all_predictions = []
    all_ground_truths = []

    # 3. 遍历每个指令类型，处理所有数据集
    model_name_safe = model_name.replace("/", "_").replace("\\", "_")

    # 按指令类型处理
    for instruction_name in sorted(instruction_dirs.keys()):
        instruction_paths = instruction_dirs[instruction_name]

        logger.info(f"\n{'='*60}")
        logger.info(
            f"处理指令类型: {instruction_name} (共 {len(instruction_paths)} 个数据集)")
        logger.info(f"{'='*60}")

        instruction_predictions = []
        instruction_ground_truths = []

        # 处理该指令类型下的所有数据集
        for dataset_idx, (dataset_name, dataset_path, instruction_path) in enumerate(instruction_paths):
            logger.info(f"\n{'='*40}")
            logger.info(
                f"处理数据集 [{dataset_idx+1}/{len(instruction_paths)}]: {dataset_name}/{instruction_name}")
            logger.info(f"数据集路径: {dataset_path}")
            logger.info(f"指令路径: {instruction_path}")
            logger.info(f"{'='*40}")

            video_dir = instruction_path
            meta_file = os.path.join(instruction_path, "annotations.json")

            # 检查文件是否存在
            if not os.path.exists(meta_file):
                logger.error(f"标注文件不存在: {meta_file}，跳过该数据集")
                continue

            if not os.path.exists(video_dir):
                logger.error(f"视频目录不存在: {video_dir}，跳过该数据集")
                continue

            # 创建对应的输出目录，按指令类型组织
            output_dir = os.path.join(base_output_dir, instruction_name)
            os.makedirs(output_dir, exist_ok=True)

            # 为每个数据集单独保存结果文件（用于追踪）
            dataset_results_file = os.path.join(
                output_dir, f"results_{dataset_name}_{model_name_safe}.json")

            # 检查是否已有结果文件
            has_results = os.path.exists(dataset_results_file)
            logger.info(f"结果文件路径: {dataset_results_file}")
            logger.info(f"结果文件是否存在: {has_results}")

            predictions = []
            ground_truths = []
            skip_inference = False

            if has_results:
                # 结果文件已存在，加载它
                try:
                    with open(dataset_results_file, 'r', encoding='utf-8') as f:
                        results_data = json.load(f)
                        predictions = results_data.get("predictions", [])
                        ground_truths = results_data.get("ground_truths", [])
                        # 确保 ground_truths 中包含 _video_dir 字段（用于评估时定位视频）
                        for gt in ground_truths:
                            if "_video_dir" not in gt:
                                gt["_video_dir"] = video_dir
                        logger.info(
                            f"✓ 已加载 {dataset_name}/{instruction_name} 的已有结果，共 {len(predictions)} 个样本")
                        skip_inference = True
                except Exception as e:
                    logger.warning(
                        f"✗ 加载 {dataset_name}/{instruction_name} 的结果文件失败: {e}，将重新推理")
                    skip_inference = False

            if not skip_inference:
                # 需要重新推理
                logger.info(f"开始推理 {dataset_name}/{instruction_name}...")
                try:
                    predictions, ground_truths = process_single_directory(
                        video_dir, meta_file, output_dir, model, logger
                    )

                    # 保存推理结果（按数据集单独保存）
                    if predictions:
                        results_data = {
                            "model_name": model_name,
                            "model_provider": model_provider,
                            "dataset": dataset_name,
                            "instruction": instruction_name,
                            "predictions": predictions,
                            "ground_truths": ground_truths
                        }
                        with open(dataset_results_file, "w", encoding="utf-8") as f:
                            json.dump(results_data, f, indent=2,
                                      ensure_ascii=False)
                        logger.info(
                            f"✓ {dataset_name}/{instruction_name} 的推理结果已保存至: {dataset_results_file}")
                    else:
                        logger.warning(
                            f"✗ {dataset_name}/{instruction_name} 推理后没有有效结果")
                except Exception as e:
                    logger.error(
                        f"✗ 处理 {dataset_name}/{instruction_name} 时发生错误: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue

            # 累积到指令级别的结果
            if predictions:
                instruction_predictions.extend(predictions)
                instruction_ground_truths.extend(ground_truths)
                logger.info(
                    f"✓ {dataset_name}/{instruction_name} 已累积到指令级别，当前指令总样本数: {len(instruction_predictions)}")
            else:
                logger.warning(
                    f"✗ {dataset_name}/{instruction_name} 没有有效结果，无法累积")

        # 保存该指令类型下所有数据集的合并结果
        logger.info(f"\n指令 {instruction_name} 处理完成:")
        logger.info(f"  处理的数据集数量: {len(instruction_paths)}")
        logger.info(f"  累积的样本数量: {len(instruction_predictions)}")
        processed_datasets = [d[0] for d in instruction_paths]
        logger.info(f"  处理的数据集: {processed_datasets}")

        if len(instruction_predictions) == 0:
            logger.warning(f"⚠️  警告：指令 {instruction_name} 没有累积到任何样本！")

        if instruction_predictions:
            all_predictions_by_instruction[instruction_name] = instruction_predictions
            all_ground_truths_by_instruction[instruction_name] = instruction_ground_truths
            all_predictions.extend(instruction_predictions)
            all_ground_truths.extend(instruction_ground_truths)

            # 评估该指令类型（合并所有数据集）
            logger.info(
                f"\n开始计算指令 {instruction_name} 的合并评估指标... (共 {len(instruction_predictions)} 个样本，使用 {Config.EVAL_NUM_WORKERS} 个线程并行评估)")
            instruction_metrics = Evaluator.evaluate_batch(
                instruction_predictions, instruction_ground_truths, num_workers=Config.EVAL_NUM_WORKERS)

            logger.info(f"指令 {instruction_name} 的评估结果:")
            logger.info(json.dumps(instruction_metrics, indent=2))

            # 保存指令级别的评估结果
            instruction_metrics_path = os.path.join(
                base_output_dir, instruction_name, "metrics.json")
            with open(instruction_metrics_path, "w", encoding="utf-8") as f:
                json.dump(instruction_metrics, f, indent=2, ensure_ascii=False)
            logger.info(
                f"指令 {instruction_name} 的评估结果已保存至: {instruction_metrics_path}")
        else:
            logger.warning(f"指令 {instruction_name} 没有有效的预测结果，无法进行评估。")

    # 4. 保存所有目录的汇总结果
    if all_predictions:
        logger.info(f"\n{'='*60}")
        logger.info("保存所有目录的汇总结果...")
        logger.info(f"{'='*60}")

        summary_results_file = os.path.join(
            base_output_dir, f"results_{model_name_safe}_summary.json")
        summary_results_data = {
            "model_name": model_name,
            "model_provider": model_provider,
            "data_root_dir": data_root_dir,
            "processed_datasets": [d[0] for d in dataset_dirs],
            "processed_instructions": sorted(instruction_dirs.keys()),
            "total_samples": len(all_predictions),
            "predictions": all_predictions,
            "ground_truths": all_ground_truths
        }
        with open(summary_results_file, "w", encoding="utf-8") as f:
            json.dump(summary_results_data, f, indent=2, ensure_ascii=False)
        logger.info(f"汇总推理结果已保存至: {summary_results_file}")

        # 评估所有目录的汇总结果
        logger.info(
            f"\n开始计算所有数据的汇总评估指标... (共 {len(all_predictions)} 个样本，使用 {Config.EVAL_NUM_WORKERS} 个线程并行评估)")
        summary_metrics = Evaluator.evaluate_batch(
            all_predictions, all_ground_truths, num_workers=Config.EVAL_NUM_WORKERS)

        # 收集各指令的评估结果
        instruction_metrics_dict = {}
        for inst_name in sorted(all_predictions_by_instruction.keys()):
            inst_metrics_path = os.path.join(
                base_output_dir, inst_name, "metrics.json")
            if os.path.exists(inst_metrics_path):
                try:
                    with open(inst_metrics_path, 'r', encoding='utf-8') as f:
                        instruction_metrics_dict[inst_name] = json.load(f)
                except Exception as e:
                    logger.warning(f"加载指令 {inst_name} 的评估结果失败: {e}")

        # 构建完整的汇总评估结果
        summary_metrics_complete = {
            "overall": summary_metrics,
            "by_instruction": instruction_metrics_dict,
            "statistics": {
                "total_samples": len(all_predictions),
                "samples_by_instruction": {
                    inst: len(preds) for inst, preds in all_predictions_by_instruction.items()
                }
            }
        }

        logger.info("汇总评估结果:")
        logger.info(json.dumps(summary_metrics_complete, indent=2))

        # 保存汇总评估结果到文件
        summary_metrics_path = os.path.join(
            base_output_dir, "metrics_summary.json")
        with open(summary_metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary_metrics_complete, f,
                      indent=2, ensure_ascii=False)
        logger.info(f"汇总评估结果已保存至: {summary_metrics_path}")
    else:
        logger.warning("没有有效的预测结果，无法进行评估。")

    logger.info("任务结束")


if __name__ == "__main__":
    main()
