# main.py
"""
VSIG任务主程序
逻辑流程：
1. 初始化模型
2. 扫描数据根目录，找到所有指令文件夹
3. 对每个指令文件夹：
   a. 读取视频、annotation、description（使用DataLoader）
   b. 格式化GT数据（使用GTFormatter）
   c. 对每个样本进行推理
   d. 评估结果
"""
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
from src.data_loader import DataLoader
from src.gt_formatter import GTFormatter


def process_single_sample(formatted_gt, options_text, output_dir, model, logger, idx, total, model_config=None):
    """
    处理单个视频样本

    Args:
        formatted_gt: 已格式化的GT数据（包含所有必要信息）
        options_text: 选项定义文本（用于构建prompt）
        output_dir: 结果保存目录
        model: 模型实例
        logger: 日志记录器
        idx: 当前样本索引（从0开始）
        total: 总样本数
        model_config: 模型配置字典（可选，用于获取模型特定的配置）

    Returns:
        tuple: (prediction, ground_truth) 或 (None, None) 如果处理失败
    """
    video_id = formatted_gt["video_name"]
    video_dir = formatted_gt["_video_dir"]
    logger.info(f"[{idx+1}/{total}] 处理样本: {video_id}")

    video_path = os.path.join(video_dir, video_id)

    # 获取 ASR 文本，如果没有则使用任务模板描述
    asr_result = formatted_gt.get("asr_result")
    transcript = asr_result["text"] if (
        asr_result and isinstance(asr_result, dict) and "text" in asr_result) else formatted_gt.get("task_template")

    if formatted_gt.get("task_template") == "指令1":
        transcript = "用户没有说话，只是做出了指向性动作。"

    # 获取模型配置（优先使用传入的配置，否则使用全局配置）
    use_video_input = model_config.get(
        "use_video_input", Config.USE_VIDEO_INPUT) if model_config else Config.USE_VIDEO_INPUT
    coord_order = model_config.get(
        "coord_order", Config.COORD_ORDER) if model_config else Config.COORD_ORDER
    model_name = model_config.get(
        "name", Config.MODEL_NAME) if model_config else Config.MODEL_NAME

    # 1. 提取帧 / 准备输入
    last_frame_path = None
    frame_paths = []

    try:
        if use_video_input and hasattr(model, 'generate_from_video') and getattr(model, 'accepts_video_files', False):
            logger.info(f"使用直接视频输入模式: {video_path}")
            # 视频输入模式：仅提取最后一帧用于可视化
            try:
                _, last_frame_path = VideoProcessor.extract_frame(
                    video_path, timestamp_sec=None)
            except Exception as vid_e:
                logger.warning(f"无法提取可视化帧 (非致命错误): {vid_e}")
        else:
            if use_video_input:
                logger.warning(f"配置了视频输入但模型 {model_name} 不支持，回退到抽帧模式")

            # 抽帧模式：需要提取多帧用于推理
            frame_paths, last_frame_path = VideoProcessor.extract_frames(
                video_path, num_frames=Config.NUM_FRAMES, end_timestamp_sec=formatted_gt.get("timestamp"))
    except Exception as e:
        logger.error(f"处理视频 {video_id} 失败: {e}")
        return None, None

    # 2. 构建 Prompt
    system_prompt = VSIGPrompts.get_system_prompt(
        task_template=formatted_gt.get("task_template"),
        coord_order=coord_order,
        options_text=options_text
    )
    user_prompt = VSIGPrompts.get_user_prompt(
        transcript, asr_result=asr_result)
    logger.info(f"system_prompt: {system_prompt}")
    logger.info(f"user_prompt: {user_prompt}")
    # 3. 模型推理
    try:
        if use_video_input and hasattr(model, 'generate_from_video') and getattr(model, 'accepts_video_files', False):
            result = model.generate_from_video(
                video_path, user_prompt, system_prompt=system_prompt)
        else:
            result = model.generate(
                frame_paths, user_prompt, system_prompt=system_prompt)
        logger.info(f"result: {result}")
    except Exception as e:
        logger.error(f"样本 {video_id} 推理出错: {e}")
        return None, None

    if not result:
        logger.warning(f"样本 {video_id} 推理无结果，跳过")
        return None, None

    # 4. 处理 result 格式
    if isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], dict):
            result = result[0]
            logger.warning(f"样本 {video_id} 返回结果为列表格式，已提取第一个元素")
        else:
            logger.warning(f"样本 {video_id} 返回结果为列表但格式不正确，跳过")
            return None, None

    if not isinstance(result, dict):
        logger.warning(f"样本 {video_id} 返回结果格式不正确: {type(result)}，跳过")
        return None, None

    result["video_name"] = video_id

    # 注意：坐标转换已在模型类的 generate/generate_from_video 方法中完成
    # result["point_list"] 中的所有 points 都已经是 [x, y] 格式

    # 5. 可视化
    vis_path = os.path.join(output_dir, f"vis_{video_id}.jpg")
    try:
        # 使用已格式化的GT数据中的processed_gt
        processed_gt = formatted_gt.get("_processed_gt", {})
        gt_items = processed_gt.get("items", [])
        VideoProcessor.visualize_points(
            last_frame_path, result, vis_path, gt_json=formatted_gt, gt_items=gt_items)
    except Exception as e:
        logger.error(f"样本 {video_id} 可视化失败: {e}")

    explicit_cmd = result.get('explicit_command', 'None')
    logger.info(f"样本 {video_id} 完成。指令: {explicit_cmd}")

    return result, formatted_gt


def process_single_directory(video_dir, meta_file, output_dir, model, logger, model_config=None):
    """
    处理单个指令文件夹

    统一的数据加载和GT格式化流程：
    1. 使用 DataLoader.prepare_dataset 读取数据
    2. 立即使用 GTFormatter.format_batch_gt_for_evaluation 格式化所有GT
    3. 对每个样本进行推理

    Args:
        video_dir: 视频存放目录
        meta_file: 标注文件路径
        output_dir: 结果保存目录
        model: 模型实例
        logger: 日志记录器
        model_config: 模型配置字典（可选，用于传递给process_single_sample）

    Returns:
        predictions: 预测结果列表
        ground_truths: 真实标签列表（已格式化）
    """
    # 1. 读取数据：视频、annotation、eval_gt.json（使用DataLoader）
    dataset, options_text, video_eval_data = DataLoader.prepare_dataset(
        video_dir=video_dir,
        annotation_path=meta_file
    )

    logger.info(f"成功加载数据集，共 {len(dataset)} 条样本")

    # 2. 格式化GT数据（使用GTFormatter）
    formatted_gt_list = GTFormatter.format_batch_gt_for_evaluation(
        dataset, video_dir, video_eval_data
    )

    # 如果 video_eval_data 不为空，说明会过滤掉不在评估列表中的视频
    if video_eval_data and len(video_eval_data) > 0:
        skipped_count = len(dataset) - len(formatted_gt_list)
        if skipped_count > 0:
            logger.info(
                f"已过滤掉 {skipped_count} 个不符合标准的视频（不在 eval_gt.json 中）")

    logger.info(f"格式化后剩余 {len(formatted_gt_list)} 条样本用于评估")

    predictions = []
    ground_truths = []
    os.makedirs(output_dir, exist_ok=True)

    # 3. 推理循环（支持多线程并行）
    logger.info(f"开始推理循环... (使用 {Config.NUM_WORKERS} 个线程并行处理)")

    if Config.NUM_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
            future_to_gt = {
                executor.submit(
                    process_single_sample,
                    formatted_gt,
                    options_text,
                    output_dir,
                    model,
                    logger,
                    idx,
                    len(formatted_gt_list),
                    model_config
                ): formatted_gt
                for idx, formatted_gt in enumerate(formatted_gt_list)
            }

            for future in as_completed(future_to_gt):
                try:
                    prediction, ground_truth = future.result()
                    if prediction is not None and ground_truth is not None:
                        predictions.append(prediction)
                        ground_truths.append(ground_truth)
                except Exception as e:
                    gt = future_to_gt[future]
                    logger.error(
                        f"处理样本 {gt.get('video_name', 'unknown')} 时发生异常: {e}")
    else:
        for idx, formatted_gt in enumerate(formatted_gt_list):
            prediction, ground_truth = process_single_sample(
                formatted_gt, options_text, output_dir, model, logger, idx, len(formatted_gt_list), model_config)
            if prediction is not None and ground_truth is not None:
                predictions.append(prediction)
                ground_truths.append(ground_truth)

    logger.info(f"推理完成，成功处理 {len(predictions)}/{len(formatted_gt_list)} 个样本")

    return predictions, ground_truths


def initialize_model(model_config, logger):
    """
    根据模型配置初始化模型实例

    Args:
        model_config: 模型配置字典，包含 provider, name, api_key 等字段
        logger: 日志记录器

    Returns:
        模型实例
    """
    provider = model_config.get("provider")
    model_name = model_config.get("name")

    # 获取API密钥（优先使用模型配置中的，否则使用全局配置）
    if provider == "openai":
        api_key = model_config.get("api_key") or Config.OPENAI_API_KEY
        base_url = model_config.get("base_url") or Config.OPENAI_BASE_URL

        if not api_key:
            logger.error(
                f"未找到 OpenAI API Key（模型: {model_name}）。请在 config.py 中配置 OPENAI_API_KEY 或设置环境变量。")
            return None

        logger.info(f"正在初始化 OpenAIVLM (Model: {model_name})")
        # 获取视频输入配置（优先使用模型配置中的，否则使用全局配置）
        use_video_input = model_config.get(
            "use_video_input", Config.USE_VIDEO_INPUT)
        coord_order = model_config.get("coord_order", Config.COORD_ORDER)
        model = OpenAIVLM(api_key=api_key, base_url=base_url,
                          model_name=model_name, accepts_video_files=use_video_input,
                          coord_order=coord_order)

    elif provider == "gemini":
        api_key = model_config.get("api_key") or Config.GEMINI_API_KEY

        if not api_key:
            logger.error(
                f"未找到 Gemini API Key（模型: {model_name}）。请在 config.py 中配置 GEMINI_API_KEY 或设置环境变量。")
            return None

        logger.info(f"正在初始化 GeminiVLM (Model: {model_name})")
        coord_order = model_config.get("coord_order", Config.COORD_ORDER)
        model = GeminiVLM(api_key=api_key, model_name=model_name,
                          coord_order=coord_order)

    else:
        logger.error(f"不支持的模型提供商: {provider}")
        return None

    return model


def process_single_model(model_config, logger):
    """
    处理单个模型的完整流程

    Args:
        model_config: 模型配置字典
        logger: 日志记录器
    """
    provider = model_config.get("provider")
    model_name = model_config.get("name")

    logger.info(f"\n{'='*80}")
    logger.info(f"开始处理模型: {model_name} ({provider})")
    logger.info(f"{'='*80}")

    # 1. 初始化模型
    model = initialize_model(model_config, logger)
    if model is None:
        logger.error(f"模型 {model_name} 初始化失败，跳过")
        return

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
    # 使用模型名称构建输出目录
    model_name_safe = model_name.replace("/", "_").replace("\\", "_")
    
    # 优先从 Config.OUTPUT_DIR 获取根目录
    results_root = "results"
    if Config.OUTPUT_DIR:
        if "/" in Config.OUTPUT_DIR or "\\" in Config.OUTPUT_DIR:
            results_root = os.path.dirname(Config.OUTPUT_DIR)
        else:
            results_root = Config.OUTPUT_DIR
    
    base_output_dir = os.path.join(results_root, model_name_safe)

    if not os.path.exists(data_root_dir):
        logger.critical(f"数据根目录不存在: {data_root_dir}")
        sys.exit(1)

    # 2. 扫描数据根目录，找到所有指令文件夹（使用DataLoader）
    instruction_dirs = DataLoader.scan_data_root(data_root_dir)

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
    logger.info(f"{'='*60}\n")

    # 存储所有目录的预测结果和真实标签（按指令分组）
    all_predictions_by_instruction = {}  # {指令名: [predictions]}
    all_ground_truths_by_instruction = {}  # {指令名: [ground_truths]}
    all_predictions = []
    all_ground_truths = []

    # 3. 遍历每个指令类型，处理所有数据集
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
                        video_dir, meta_file, output_dir, model, logger, model_config
                    )

                    # 保存推理结果（按数据集单独保存）
                    if predictions:
                        results_data = {
                            "model_name": model_name,
                            "model_provider": provider,
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
        else:
            logger.warning(f"指令 {instruction_name} 没有有效的预测结果，无法进行评估。")

    # 4. 保存所有目录的汇总结果
    if all_predictions:
        logger.info(f"\n{'='*60}")
        logger.info("保存所有目录的汇总结果...")
        logger.info(f"{'='*60}")

        summary_results_file = os.path.join(
            base_output_dir, f"results_{model_name_safe}_summary.json")
        # 从instruction_dirs中提取所有数据集名称
        all_datasets = set()
        for dirs_list in instruction_dirs.values():
            for dataset_name, _, _ in dirs_list:
                all_datasets.add(dataset_name)

        summary_results_data = {
            "model_name": model_name,
            "model_provider": provider,
            "data_root_dir": data_root_dir,
            "processed_datasets": sorted(all_datasets),
            "processed_instructions": sorted(instruction_dirs.keys()),
            "total_samples": len(all_predictions),
            "predictions": all_predictions,
            "ground_truths": all_ground_truths
        }
        with open(summary_results_file, "w", encoding="utf-8") as f:
            json.dump(summary_results_data, f, indent=2, ensure_ascii=False)
        logger.info(f"汇总推理结果已保存至: {summary_results_file}")

        # 统一评估：只评估一次总体结果，然后从 instruction_breakdown 中提取各指令的评估结果
        logger.info(f"\n{'='*60}")
        logger.info("开始统一评估所有结果...")
        logger.info(f"{'='*60}")

        # 评估总体结果（一次评估即可，避免重复评估）
        logger.info(
            f"\n开始计算所有数据的汇总评估指标... (共 {len(all_predictions)} 个样本，使用 {Config.EVAL_NUM_WORKERS} 个线程并行评估)")
        summary_metrics = Evaluator.evaluate_batch(
            all_predictions, all_ground_truths, num_workers=Config.EVAL_NUM_WORKERS)

        # 从总体评估结果的 instruction_breakdown 中提取各指令的评估结果（避免重复评估）
        instruction_breakdown = summary_metrics.get(
            "instruction_breakdown", {})
        instruction_metrics_dict = {}

        # 为每个指令构建完整的评估结果格式（与单独评估时的格式保持一致）
        for inst_name in sorted(all_predictions_by_instruction.keys()):
            if inst_name in instruction_breakdown:
                # 从 instruction_breakdown 中提取该指令的评估结果
                inst_breakdown = instruction_breakdown[inst_name]

                # 构建与单独评估时相同格式的评估结果
                inst_metrics = {
                    "overall_accuracy": inst_breakdown.get("intent_accuracy", 0.0),
                    "intent_grounding_accuracy": inst_breakdown.get("intent_accuracy", 0.0),
                    "spatial_grounding_accuracy": inst_breakdown.get("spatial_grounding", 0.0),
                    "speech_temporal_grounding_accuracy": inst_breakdown.get("speech_temporal_grounding", 0.0),
                    "overall_score": inst_breakdown.get("overall", 0.0),
                    "instruction_breakdown": {inst_name: inst_breakdown},
                    "count": inst_breakdown.get("count", 0)
                }

                instruction_metrics_dict[inst_name] = inst_metrics

                logger.info(f"指令 {inst_name} 的评估结果:")
                logger.info(json.dumps(inst_metrics, indent=2))

                # 保存指令级别的评估结果
                instruction_metrics_path = os.path.join(
                    base_output_dir, inst_name, "metrics.json")
                with open(instruction_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(inst_metrics, f, indent=2, ensure_ascii=False)
                logger.info(
                    f"指令 {inst_name} 的评估结果已保存至: {instruction_metrics_path}")
            else:
                logger.warning(f"警告：指令 {inst_name} 在评估结果中未找到，可能没有对应的样本")

        # 3. 构建完整的汇总评估结果
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

        # 保存汇总评估结果到文件 (包含详细样本结果)
        summary_metrics_path = os.path.join(
            base_output_dir, "metrics_summary.json")
        with open(summary_metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary_metrics_complete, f,
                      indent=2, ensure_ascii=False)
        logger.info(f"汇总评估结果已保存至: {summary_metrics_path}")

        # 打印汇总评估结果 (不包含详细结果，避免终端输出过长)
        if "overall" in summary_metrics_complete and "detailed_results" in summary_metrics_complete["overall"]:
            del summary_metrics_complete["overall"]["detailed_results"]

        logger.info("汇总评估结果 (不包含详细样本结果):")
        logger.info(json.dumps(summary_metrics_complete, indent=2))
    else:
        logger.warning("没有有效的预测结果，无法进行评估。")

    logger.info(f"模型 {model_name} 处理完成")
    logger.info(f"{'='*80}\n")


def main():
    # 0. 初始化 Logger
    # 获取结果根目录
    results_root = "results"
    if Config.OUTPUT_DIR:
        if "/" in Config.OUTPUT_DIR or "\\" in Config.OUTPUT_DIR:
            results_root = os.path.dirname(Config.OUTPUT_DIR)
        else:
            results_root = Config.OUTPUT_DIR
    
    # 创建日志目录：如果配置了多模型，使用 results_root/logs；否则使用 base_output_dir/logs
    if hasattr(Config, 'MODELS') and Config.MODELS and len(Config.MODELS) > 0:
        log_dir = os.path.join(results_root, "logs")
    else:
        model_name_safe = Config.MODEL_NAME.replace(
            "/", "_").replace("\\", "_")
        log_dir = os.path.join(results_root, model_name_safe, "logs")

    logger, log_file = setup_logger(output_dir=log_dir, log_to_file=True)
    if log_file:
        logger.info(f"日志文件: {log_file}")
    logger.info("Visual-Speech Intent Grounding (VSIG) 任务启动")
    logger.info("加载配置...")

    # 检查是否配置了多模型模式
    if hasattr(Config, 'MODELS') and Config.MODELS and len(Config.MODELS) > 0:
        # 多模型模式：遍历所有配置的模型
        logger.info(f"\n{'='*80}")
        logger.info(f"检测到多模型配置，共 {len(Config.MODELS)} 个模型")
        logger.info(f"{'='*80}\n")

        for model_idx, model_config in enumerate(Config.MODELS):
            logger.info(f"\n{'#'*80}")
            logger.info(f"处理模型 [{model_idx+1}/{len(Config.MODELS)}]")
            logger.info(f"{'#'*80}")

            try:
                process_single_model(model_config, logger)
            except Exception as e:
                logger.error(
                    f"处理模型 {model_config.get('name', 'unknown')} 时发生错误: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.warning(
                    f"跳过模型 {model_config.get('name', 'unknown')}，继续处理下一个模型")
                continue

        logger.info(f"\n{'='*80}")
        logger.info("所有模型处理完成")
        logger.info(f"{'='*80}")
    else:
        # 单模型模式：使用原有的单个模型配置（保持向后兼容）
        logger.info("使用单模型配置模式")

        # 构建单个模型配置
        model_config = {
            "provider": Config.MODEL_PROVIDER,
            "name": Config.MODEL_NAME,
            "api_key": None,  # 使用全局配置
            "base_url": None,  # 使用全局配置
            "coord_order": Config.COORD_ORDER,
            "use_video_input": Config.USE_VIDEO_INPUT
        }

        try:
            process_single_model(model_config, logger)
        except Exception as e:
            logger.error(f"处理模型时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)

    logger.info("所有任务结束")


if __name__ == "__main__":
    main()
