# src/eval/metrics.py
import json
import math
import numpy as np
import base64
import cv2
import os
import re
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.gt_formatter import GTFormatter


class Evaluator:
    """
    计算 VSIG 任务的评估指标（结合 MCQ 选项和坐标预测）。
    同时评估选项选择、空间定位（point）和时间定位（timestamp）。
    """

    _eval_model = None  # 保留接口，但在 MCQ 模式下可能不需要

    @staticmethod
    def set_eval_model(model):
        """保留接口，但在 MCQ 模式下可能不需要"""
        Evaluator._eval_model = model

    @staticmethod
    def _ensure_single_point(pt):
        """辅助方法：确保输入为单点 [x, y]（取平均值）"""
        if not pt or not isinstance(pt, list):
            return None
        if len(pt) == 0:
            return None
        # 如果是 [[x, y], ...] 格式
        if isinstance(pt[0], list):
            avg_x = sum(p[0] for p in pt) / len(pt)
            avg_y = sum(p[1] for p in pt) / len(pt)
            return [avg_x, avg_y]
        # 如果是 [x, y] 格式
        return pt

    @staticmethod
    def calculate_distance(pred_pt, gt_pt):
        """
        计算预测点与 GT 点之间的欧氏距离。
        输入均应为 [x, y] 格式。
        """
        p_pt = Evaluator._ensure_single_point(pred_pt)
        g_pt = Evaluator._ensure_single_point(gt_pt)

        if p_pt is None or g_pt is None:
            return float('inf')

        return math.sqrt((p_pt[0] - g_pt[0])**2 + (p_pt[1] - g_pt[1])**2)

    @staticmethod
    def is_point_in_mask(point, mask_base64, bbox, width=1920, height=1080):
        """
        严谨的 mask 评估：检查点是否落在解码后的 mask 像素内。
        point: [x, y] 或 [[x, y], ...] (像素坐标)
        mask_base64: 标注的 mask 数据 (Base64 编码的图像)
        bbox: [x1, y1, x2, y2] (原始像素坐标)
        width, height: 视频帧的原始宽高（用于mask大小判断）
        """
        if not mask_base64 or not bbox:
            return False

        p_pt = Evaluator._ensure_single_point(point)
        if p_pt is None:
            return False

        try:
            # point 已经是像素坐标，直接使用
            px, py = int(p_pt[0]), int(p_pt[1])

            # 基础检查：是否在 bbox 内 (bbox 为 [x1, y1, x2, y2])
            x1, y1, x2, y2 = bbox
            if not (x1 <= px <= x2 and y1 <= py <= y2):
                return False

            # 解码 mask 图像
            mask_data = base64.b64decode(mask_base64)
            nparr = np.frombuffer(mask_data, np.uint8)
            mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                return False

            # 判断点在 mask 中的位置
            # 如果 mask 是全图大小 (与 width, height 一致)
            if mask.shape[0] == height and mask.shape[1] == width:
                return mask[py, px] > 128

            # 如果 mask 只是 bbox 大小的切片
            mask_h, mask_w = mask.shape
            bbox_h, bbox_w = y2 - y1, x2 - x1

            # 计算点相对于 bbox 的局部坐标
            local_y = int((py - y1) * mask_h / bbox_h) if bbox_h > 0 else 0
            local_x = int((px - x1) * mask_w / bbox_w) if bbox_w > 0 else 0

            # 边界检查
            local_y = min(max(0, local_y), mask_h - 1)
            local_x = min(max(0, local_x), mask_w - 1)

            return mask[local_y, local_x] > 128
        except Exception:
            return False

    @staticmethod
    def evaluate_temporal(pred_ts, asr_result, target_description):
        """
        让大模型评估时间定位。
        pred_ts: [start_ms, end_ms]
        asr_result: ASR 结果字典 (包含 text 和 words)
        target_description: 目标描述
        """
        if not pred_ts or not asr_result:
            return 0.0

        if not Evaluator._eval_model:
            # 如果没有评估模型，回退到简单的 IoU 计算
            return Evaluator._evaluate_temporal_fallback(pred_ts, asr_result)

        asr_text = asr_result.get("text", "")
        asr_words = asr_result.get("words", [])

        words_info = []
        for w in asr_words:
            words_info.append(
                f"'{w['text']}' ({w.get('begin_time', 0)}ms-{w.get('end_time', 0)}ms)")

        words_timeline = ", ".join(words_info)

        prompt = f"""你是一个评估视觉语言模型表现的专家助手。
用户的原始语音转录内容为："{asr_text}"。
其中每个词及其对应的时间轴（毫秒）如下：{words_timeline}。

模型预测的与"{target_description}"动作相关的视频时间范围为：{pred_ts[0]}ms 到 {pred_ts[1]}ms。

请根据语音时间轴判断，模型预测的时间范围是否准确涵盖了用户提到该物体或执行该动作时的语音段？
评分标准：
- 1.0: 预测非常准确，精准涵盖了相关的语音。
- 0.5: 预测部分重叠，或者范围过大/过小，但包含关键部分。
- 0.0: 预测完全错误，没有涵盖相关的语音段。

请以 JSON 格式返回评分结果，包含字段 "score" (0.0 到 1.0 之间的浮点数)。只返回 JSON 即可。"""

        try:
            # 调用大模型（传入空图像列表）
            result = Evaluator._eval_model.generate([], prompt)
            score = result.get("score", 0.0)
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            return max(0.0, min(1.0, score))
        except Exception:
            # 出错时回退
            return Evaluator._evaluate_temporal_fallback(pred_ts, asr_result)

    @staticmethod
    def _evaluate_temporal_fallback(pred_ts, asr_result):
        """
        传统的 IoU 时间定位评估。
        """
        pred_start, pred_end = pred_ts
        asr_words = asr_result.get("words", [])
        if not asr_words:
            return 0.0

        asr_start = min(w.get("begin_time", 0) for w in asr_words)
        asr_end = max(w.get("end_time", 0) for w in asr_words)

        intersection = max(0, min(pred_end, asr_end) -
                           max(pred_start, asr_start))
        union = max(pred_end, asr_end) - min(pred_start, asr_start)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def evaluate_intent(selected_norm, correct_norm):
        """
        评估意图准确性（基于 MCQ 选项）。
        不再是严格全匹配，而是计算位置对应的匹配项比例。
        例如：pred [A, b, C] vs gt [A, b, D] -> 得分 2/3
        """
        if not correct_norm:
            return 0.0, False

        # 计算相同位置匹配的数量
        matches = 0
        for i in range(min(len(selected_norm), len(correct_norm))):
            if selected_norm[i] == correct_norm[i]:
                matches += 1

        # 准确度 = 匹配数 / GT总数
        accuracy = matches / len(correct_norm)
        # 只有完全一致时 is_match 才为 True
        is_match = (selected_norm == correct_norm)

        return accuracy, is_match

    @staticmethod
    def _normalize_pred_to_pixel(pred_pt, width: int, height: int):
        """
        将预测点的归一化坐标（0-1000）转换为像素坐标。

        Args:
            pred_pt: 归一化坐标 [x, y] 或 [[x, y], ...]
            width: 视频宽度
            height: 视频高度

        Returns:
            像素坐标 [x, y]
        """
        if isinstance(pred_pt[0], list):
            # 多个点，取平均
            avg_x = sum(p[0] for p in pred_pt) / len(pred_pt)
            avg_y = sum(p[1] for p in pred_pt) / len(pred_pt)
        else:
            avg_x, avg_y = pred_pt[0], pred_pt[1]

        # 转换为像素坐标
        px = int(avg_x * width / 1000)
        py = int(avg_y * height / 1000)
        return [px, py]

    @staticmethod
    def evaluate_sample(pred, formatted_gt):
        """
        评估单个样本。所有信息都已预处理，直接使用即可。

        Args:
            pred (dict): 模型预测结果 (JSON)，坐标已统一为 [x, y] 格式（归一化 0-1000）
            formatted_gt (dict): 已格式化的GT数据（通过 GTFormatter.format_gt_for_evaluation 处理）
                - 包含 _video_dir, _video_width, _video_height, _last_frame_path
                - _processed_gt: GT items（points 已转换为像素坐标）
                - _correct_options: 正确答案选项列表
        """
        video_name = formatted_gt.get("video_name")
        if not video_name:
            return {
                "intent_accuracy": 0.0,
                "spatial_grounding": 0.0,
                "temporal_grounding": 0.0,
                "match": False,
                "pred_options": [],
                "gt_options": []
            }

        # 获取预处理好的视频信息
        width = formatted_gt.get("_video_width", 1920)
        height = formatted_gt.get("_video_height", 1080)

        # 获取正确答案（从已格式化的GT中提取）
        correct_answers = formatted_gt.get("_correct_options", [])

        # 获取预测答案
        selected = pred.get("selected_options", [])
        if isinstance(selected, str):
            selected = [selected]
        elif not isinstance(selected, list):
            selected = []

        # 归一化比较 (去除空格，字符串化)
        selected_norm = [str(s).strip() for s in selected]
        correct_norm = [str(a).strip() for a in correct_answers]

        # 评估意图准确性
        intent_accuracy, is_match = Evaluator.evaluate_intent(
            selected_norm, correct_norm)

        # 使用已格式化的GT数据
        processed_gt = formatted_gt.get("_processed_gt", {})
        pred_points = pred.get("point_list", [])
        asr_result = formatted_gt.get("asr_result")

        # 计算 spatial_grounding：基于point坐标的评估
        gt_items = processed_gt.get("items", [])
        spatial_scores = []

        for i, gt_item in enumerate(gt_items):
            if i >= len(pred_points):
                spatial_scores.append(0.0)
                continue

            pred_item = pred_points[i]
            pred_pt = pred_item.get("point")

            if not pred_pt:
                spatial_scores.append(0.0)
                continue

            # 获取GT点（已经是像素坐标）
            gt_points = gt_item.get("points", [])
            if not gt_points:
                spatial_scores.append(0.0)
                continue

            # 将预测点从归一化坐标转换为像素坐标
            pred_pt_pixel = Evaluator._normalize_pred_to_pixel(
                pred_pt, width, height)

            # GT points 已经是像素坐标，取平均
            if isinstance(gt_points[0], list):
                gt_pt_avg = [
                    int(sum(p[0] for p in gt_points) / len(gt_points)),
                    int(sum(p[1] for p in gt_points) / len(gt_points))
                ]
            else:
                gt_pt_avg = [int(gt_points[0]), int(gt_points[1])]

            # 如果GT有mask，使用mask检查
            if "mask" in gt_item and gt_item["mask"].get("mask_base64"):
                is_hit = Evaluator.is_point_in_mask(
                    pred_pt_pixel,
                    gt_item["mask"].get("mask_base64"),
                    gt_item["mask"].get("bbox"),
                    width=width,
                    height=height
                )
                spatial_scores.append(1.0 if is_hit else 0.0)
            else:
                # 使用距离阈值（50像素）
                dist = Evaluator.calculate_distance(pred_pt_pixel, gt_pt_avg)
                spatial_scores.append(1.0 if dist < 50 else 0.0)

        spatial_grounding = sum(spatial_scores) / \
            len(spatial_scores) if spatial_scores else 0.0

        # 计算 temporal_grounding：让大模型判断
        temporal_scores = []

        if asr_result and asr_result.get("text") and asr_result.get("words"):
            for i, pred_item in enumerate(pred_points):
                pred_ts = pred_item.get("timestamp")
                if pred_ts is None:
                    # 如果没有timestamp（如指令1），跳过评估
                    continue

                if not isinstance(pred_ts, list) or len(pred_ts) != 2:
                    temporal_scores.append(0.0)
                    continue

                # 调用 LLM 评估或回退 IoU
                target_desc = pred_item.get("description", "目标物体")
                score = Evaluator.evaluate_temporal(
                    pred_ts, asr_result, target_desc)
                temporal_scores.append(score)

        temporal_grounding = sum(temporal_scores) / \
            len(temporal_scores) if temporal_scores else 0.0

        return {
            "intent_accuracy": intent_accuracy,
            "spatial_grounding": spatial_grounding,
            "temporal_grounding": temporal_grounding,
            "match": is_match,
            "pred_options": selected_norm,
            "gt_options": correct_norm
        }

    @staticmethod
    def _evaluate_single_sample_wrapper(args):
        """
        评估单个样本的包装函数（用于多线程）。
        """
        pred, gt = args
        return Evaluator.evaluate_sample(pred, gt)

    @staticmethod
    def evaluate_batch(predictions, ground_truths, video_dir=None, num_workers=None):
        """
        批量评估结果（支持多线程并行）。

        Args:
            predictions: 预测结果列表（坐标已统一为 [x, y] 格式）
            ground_truths: 已格式化的GT数据列表（通过 GTFormatter.format_batch_gt_for_evaluation 处理）
                - 每个GT包含 _video_dir, _processed_gt, _correct_options 等字段
            video_dir: 视频目录（可选，用于预加载视频分辨率）
            num_workers: 并行评估线程数，None 表示使用单线程
        """
        if not predictions or not ground_truths:
            return {}

        # 准备评估参数（视频信息已在GT格式化阶段预处理）
        eval_args = [(pred, gt)
                     for pred, gt in zip(predictions, ground_truths)]

        # 并行评估
        results_with_gt = []  # 存储 (score, gt, pred) 三元组，确保数据对齐
        if num_workers and num_workers > 1 and len(eval_args) > 1:
            with ThreadPoolExecutor(max_workers=min(num_workers, len(eval_args))) as executor:
                futures = {executor.submit(Evaluator._evaluate_single_sample_wrapper, args): args
                           for args in eval_args}
                for future in as_completed(futures):
                    # 从原始 args 中提取对应的 pred 和 gt
                    pred, gt = futures[future]
                    try:
                        score = future.result()
                        results_with_gt.append((score, gt, pred))
                    except Exception as e:
                        default_score = {
                            "intent_accuracy": 0.0,
                            "spatial_grounding": 0.0,
                            "temporal_grounding": 0.0,
                            "match": False,
                            "pred_options": [],
                            "gt_options": []
                        }
                        results_with_gt.append((default_score, gt, pred))
        else:
            # 单线程模式
            for args in eval_args:
                pred, gt = args
                try:
                    score = Evaluator._evaluate_single_sample_wrapper(args)
                    results_with_gt.append((score, gt, pred))
                except Exception as e:
                    default_score = {
                        "intent_accuracy": 0.0,
                        "spatial_grounding": 0.0,
                        "temporal_grounding": 0.0,
                        "match": False,
                        "pred_options": [],
                        "gt_options": []
                    }
                    results_with_gt.append((default_score, gt, pred))

        if not results_with_gt:
            return {}

        # 统一提取 scores 进行汇总计算
        all_scores = [x[0] for x in results_with_gt]

        avg_intent = sum(s["intent_accuracy"]
                         for s in all_scores) / len(all_scores)

        # 空间和时间 Grounding 取平均（现在是标量值）
        all_spatial = [s["spatial_grounding"] for s in all_scores]
        avg_spatial = sum(all_spatial) / \
            len(all_spatial) if all_spatial else 0.0

        all_temporal = [s["temporal_grounding"]
                        for s in all_scores if s.get("temporal_grounding", 0) > 0]
        avg_temporal = sum(all_temporal) / \
            len(all_temporal) if all_temporal else 0.0

        # 按指令类型统计 (Instruction-wise breakdown)
        instruction_breakdown = {}
        # NOTE: We must use results_with_gt because all_scores (from as_completed) is not in sync with eval_args
        for score, gt, pred in results_with_gt:
            instr = gt.get("task_template", "Unknown")
            if instr not in instruction_breakdown:
                instruction_breakdown[instr] = {
                    "intent_accuracy": [],
                    "spatial_grounding": [],
                    "temporal_grounding": [],
                    "count": 0
                }

            ib = instruction_breakdown[instr]
            ib["count"] += 1
            ib["intent_accuracy"].append(score["intent_accuracy"])
            ib["spatial_grounding"].append(score["spatial_grounding"])
            if score.get("temporal_grounding", 0) > 0:
                ib["temporal_grounding"].append(score["temporal_grounding"])

        breakdown_results = {}
        for instr, data in instruction_breakdown.items():
            avg_i = sum(data["intent_accuracy"]) / len(data["intent_accuracy"])
            avg_s = sum(data["spatial_grounding"]) / \
                len(data["spatial_grounding"]
                    ) if data["spatial_grounding"] else 0.0
            avg_t = sum(data["temporal_grounding"]) / \
                len(data["temporal_grounding"]
                    ) if data["temporal_grounding"] else 0.0

            breakdown_results[instr] = {
                "intent_accuracy": avg_i,
                "spatial_grounding": avg_s,
                "temporal_grounding": avg_t,
                "overall": (avg_i + avg_s + avg_t) / 3 if avg_t > 0 else (avg_i + avg_s) / 2,
                "count": data["count"]
            }

        return {
            "overall_accuracy": avg_intent,
            "intent_grounding_accuracy": avg_intent,  # 保持兼容性
            "spatial_grounding_accuracy": avg_spatial,
            "temporal_grounding_accuracy": avg_temporal,
            "overall_score": (avg_intent + avg_spatial + avg_temporal) / 3 if avg_temporal > 0 else (avg_intent + avg_spatial) / 2,
            "instruction_breakdown": breakdown_results,
            "detailed_results": [
                {
                    "video_name": gt.get("video_name"),
                    "instruction": gt.get("task_template"),
                    "pred_options": score.get("pred_options", []),
                    "gt_options": score.get("gt_options", []),
                    "match": score.get("match", False),
                    "spatial_grounding": score.get("spatial_grounding", 0.0),
                    "temporal_grounding": score.get("temporal_grounding", 0.0)
                }
                for score, gt, pred in results_with_gt
            ]
        }
