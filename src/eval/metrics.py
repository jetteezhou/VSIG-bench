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


class Evaluator:
    """
    计算 VSIG 任务的评估指标，支持 LLM-based 语义评估。
    """

    _eval_model = None  # 用于评估的 LLM 实例

    @staticmethod
    def set_eval_model(model):
        """设置用于评估的 LLM 助手"""
        Evaluator._eval_model = model

    @staticmethod
    def calculate_distance(pred_pt, gt_pt):
        """
        计算预测点与 GT 点之间的欧氏距离。
        pred_pt: [y, x]
        gt_pt: [y, x] 或 [[y1, x1], [y2, x2], ...]
        """
        if pred_pt is None or gt_pt is None:
            return float('inf')

        # 如果 GT 是多个点，取平均值
        if isinstance(gt_pt[0], list):
            gt_y = sum(p[0] for p in gt_pt) / len(gt_pt)
            gt_x = sum(p[1] for p in gt_pt) / len(gt_pt)
        else:
            gt_y, gt_x = gt_pt

        pred_y, pred_x = pred_pt
        return math.sqrt((pred_y - gt_y)**2 + (pred_x - gt_x)**2)

    @staticmethod
    def is_point_in_mask(point, mask_base64, bbox, width=1920, height=1080):
        """
        严谨的 mask 评估：检查点是否落在解码后的 mask 像素内。
        point: [y, x] (归一化 0-1000)
        mask_base64: 标注的 mask 数据 (Base64 编码的图像)
        bbox: [y1, x1, y2, x2] (原始像素坐标)
        width, height: 视频帧的原始宽高
        """
        if not mask_base64 or not bbox:
            return False

        try:
            # 1. 还原归一化坐标为像素坐标
            y_norm, x_norm = point
            py = int(y_norm * height / 1000)
            px = int(x_norm * width / 1000)

            # 2. 基础检查：是否在 bbox 内
            x1, y1, x2, y2 = bbox
            if not (y1 <= py <= y2 and x1 <= px <= x2):
                return False

            # 3. 解码 mask 图像
            mask_data = base64.b64decode(mask_base64)
            nparr = np.frombuffer(mask_data, np.uint8)
            mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                return False

            # 4. 判断点在 mask 中的位置
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
    def evaluate_temporal(pred_ts, gt_words, target_text):
        """
        评估时间定位。
        pred_ts: [start_ms, end_ms]
        gt_words: ASR 结果中的 words 列表
        target_text: 目标词汇 (如 "这个", "这里", "它的右边")
        """
        if not pred_ts or not gt_words:
            return 0.0

        # 寻找 ASR 中匹配的词语时间范围
        gt_range = None
        for word in gt_words:
            if target_text in word["text"]:
                gt_range = [word["begin_time"], word["end_time"]]
                break

        if not gt_range:
            return 0.0

        # 计算交并比 (IoU) 或检查是否重叠
        p_start, p_end = pred_ts
        g_start, g_end = gt_range

        intersection = max(0, min(p_end, g_end) - max(p_start, g_start))
        union = max(p_end, g_end) - min(p_start, g_start)

        return intersection / union if union > 0 else 0.0

    # GT 处理缓存
    _gt_cache = {}

    @staticmethod
    def process_gt_by_template(gt):
        """
        根据任务模板对 GT 数据进行预处理，特别是指令 4/5/6 的合并逻辑。
        使用缓存优化重复处理。
        """
        # 使用 GT 的 JSON 字符串作为缓存键
        gt_key = json.dumps(gt, sort_keys=True, ensure_ascii=False)

        if gt_key in Evaluator._gt_cache:
            return Evaluator._gt_cache[gt_key]

        template = gt.get("task_template")
        raw_os = gt.get("object_space", [])
        processed_gt = {
            "template": template,
            "items": []
        }

        if template in ["指令1", "指令2"]:
            # 只有一个物体
            if len(raw_os) >= 1:
                processed_gt["items"].append(raw_os[0])

        elif template == "指令3":
            # 一个物体 + 一个空间
            if len(raw_os) >= 2:
                processed_gt["items"].extend(raw_os[:2])

        elif template == "指令4":
            # 合并逻辑: [obj1, obj2, space1] -> [obj1, combined_space]
            if len(raw_os) >= 3:
                obj1 = raw_os[0]
                obj2 = raw_os[1]
                space1 = raw_os[2]
                combined_space = {
                    "name": f"{obj2['name']}的{space1['name']}",
                    "points": space1["points"],
                    "type": "space"
                }
                processed_gt["items"] = [obj1, combined_space]

        elif template == "指令5":
            # [obj1, obj2, space1, obj3] -> [obj1, combined_space, obj3]
            if len(raw_os) >= 4:
                obj1 = raw_os[0]
                obj2 = raw_os[1]
                space1 = raw_os[2]
                obj3 = raw_os[3]
                combined_space = {
                    "name": f"{obj2['name']}的{space1['name']}",
                    "points": space1["points"],
                    "type": "space"
                }
                processed_gt["items"] = [obj1, combined_space, obj3]

        elif template == "指令6":
            # [obj1, obj2, space1, obj3, obj4, space2] -> [obj1, combined_space1, obj3, combined_space2]
            if len(raw_os) >= 6:
                obj1, obj2, space1, obj3, obj4, space2 = raw_os[:6]
                combined_space1 = {
                    "name": f"{obj2['name']}的{space1['name']}",
                    "points": space1["points"],
                    "type": "space"
                }
                combined_space2 = {
                    "name": f"{obj4['name']}的{space2['name']}",
                    "points": space2["points"],
                    "type": "space"
                }
                processed_gt["items"] = [
                    obj1, combined_space1, obj3, combined_space2]

        Evaluator._gt_cache[gt_key] = processed_gt
        return processed_gt

    @staticmethod
    def llm_judge_intent(pred_command, gt_items, task_template):
        """
        使用 LLM 对意图定位进行语义评估。
        """
        if not Evaluator._eval_model:
            return 0.5

        eval_prompt = f"""你是一个严谨的机器人指令评估专家。请根据以下信息对 AI 指令系统的意图转换进行打分。

**任务场景**：{task_template}
**Ground Truth (标准意图描述)**：{[item['name'] for item in gt_items]}

**AI 系统的输出指令**："{pred_command}"

**评分准则**：
1. **意图准确性 (Intent Accuracy)**：AI 转换后的指令是否完全表达了 GT 中的意图？
2. 忽略助词差异（如"的"）、同义词差异（如"杯子"与"水杯"），关注核心动作和物体。
3. 得分范围 [0, 1]。0 代表完全错误，1 代表完美对齐。

请以 JSON 格式返回结果，必须包含 "score" 字段，格式如下：
{{
    "score": 0.8
}}

其中 score 是一个 0 到 1 之间的浮点数，表示意图准确性得分。
"""
        try:
            response = Evaluator._eval_model.generate(
                [], eval_prompt, system_prompt="你是一个具身智能意图评估专家。")

            # 处理不同类型的响应格式
            if isinstance(response, (int, float)):
                return float(response)
            elif isinstance(response, list):
                # 如果返回的是列表，尝试提取第一个数字
                if len(response) > 0:
                    if isinstance(response[0], (int, float)):
                        return float(response[0])
                    elif isinstance(response[0], str):
                        match = re.search(r"([0-9]*\.?[0-9]+)", response[0])
                        return float(match.group(1)) if match else 0.0
                return 0.0
            elif isinstance(response, dict):
                # 如果返回的是字典，尝试从常见字段中提取分数
                for key in ["intent_accuracy_score", "score", "accuracy", "value"]:
                    if key in response:
                        val = response[key]
                        if isinstance(val, (int, float)):
                            return float(val)
                        elif isinstance(val, str):
                            match = re.search(r"([0-9]*\.?[0-9]+)", val)
                            return float(match.group(1)) if match else 0.0
                # 如果没有找到常见字段，尝试提取第一个数字值
                for val in response.values():
                    if isinstance(val, (int, float)):
                        return float(val)
                    elif isinstance(val, str):
                        match = re.search(r"([0-9]*\.?[0-9]+)", val)
                        if match:
                            return float(match.group(1))
                return 0.0
            elif isinstance(response, str):
                # 如果返回的是字符串，使用正则表达式提取数字
                match = re.search(r"([0-9]*\.?[0-9]+)", response)
                return float(match.group(1)) if match else 0.0
            else:
                return 0.0
        except Exception as e:
            print(f"LLM 意图评估出错: {e}")
            return 0.0

    @staticmethod
    def llm_judge_temporal(pred_points, asr_text, task_template):
        """
        使用 LLM 对时间定位进行语义评估。
        """
        if not Evaluator._eval_model or not asr_text or asr_text == "无语音":
            return []

        eval_prompt = f"""你是一个语音时间对齐评估专家。请判断 AI 系统输出的时间戳是否与语音内容对齐。

**任务场景**：{task_template}
**用户语音转录 (ASR)**："{asr_text}"
**AI 输出的定位点及时间戳**：
{json.dumps([{"desc": p.get("description"), "ts": p.get("timestamp")} for p in pred_points], ensure_ascii=False)}

**评分准则**：
1. 判断每个定位点提供的时间戳 [start_ms, end_ms] 是否对应了语音中描述该物体或动作的时刻。
2. 例如：若语音是"把这个（2-3秒）放到这里（4-5秒）"，第一个点的时间戳应在 2000-3000ms 左右。
3. 请为每个带有时间戳的 point 分别给出得分 [0, 1]。

请以 JSON 格式返回结果，格式如下（不要包含 Markdown 标记）：
{{
    "temporal_scores": [0.0, 0.0, ...]
}}

请确保返回有效的 JSON 格式。
"""
        try:
            response = Evaluator._eval_model.generate(
                [], eval_prompt, system_prompt="你是一个语音时间对齐评估专家。")
            if isinstance(response, str):
                response = response.replace(
                    "```json", "").replace("```", "").strip()
                result = json.loads(response)
            else:
                result = response
            return result.get("temporal_scores", [])
        except Exception as e:
            print(f"LLM 时间评估出错: {e}")
            return [0.0] * len(pred_points)

    @staticmethod
    def evaluate_sample(pred, gt, width=1920, height=1080):
        """
        评估单个样本。
        """
        processed_gt = Evaluator.process_gt_by_template(gt)
        template = gt.get("task_template")
        pred_points = pred.get("point_list", [])
        asr_result = gt.get("asr_result")

        scores = {
            "intent_accuracy": 0.0,
            "spatial_grounding": [],
            "temporal_grounding": []
        }

        # 1. 意图定位评估 (始终执行)
        scores["intent_accuracy"] = Evaluator.llm_judge_intent(
            pred.get("explicit_command", ""),
            processed_gt["items"],
            template
        )

        # 2. 时间定位评估 (仅当 ASR 存在时执行)
        if asr_result and asr_result.get("text"):
            asr_text = asr_result["text"]
            scores["temporal_grounding"] = Evaluator.llm_judge_temporal(
                pred_points,
                asr_text,
                template
            )
        else:
            # 如果没有 ASR，跳过时间评估
            scores["temporal_grounding"] = None

        # 3. 空间 Grounding (物理校验)
        gt_items = processed_gt["items"]
        for i, gt_item in enumerate(gt_items):
            if i >= len(pred_points):
                scores["spatial_grounding"].append(0.0)
                continue

            pred_pt = pred_points[i].get("point")
            if not pred_pt:
                scores["spatial_grounding"].append(0.0)
                continue

            if gt_item["type"] == "object":
                if "mask" in gt_item:
                    is_hit = Evaluator.is_point_in_mask(
                        pred_pt,
                        gt_item["mask"].get("mask_base64"),
                        gt_item["mask"].get("bbox"),
                        width=width,
                        height=height
                    )
                    scores["spatial_grounding"].append(1.0 if is_hit else 0.0)
                else:
                    dist = Evaluator.calculate_distance(
                        pred_pt, gt_item["points"])
                    scores["spatial_grounding"].append(
                        1.0 if dist < 50 else 0.0)
            else:
                dist = Evaluator.calculate_distance(pred_pt, gt_item["points"])
                scores["spatial_grounding"].append(1.0 if dist < 50 else 0.0)

        return scores

    @staticmethod
    def _load_video_resolution(video_path):
        """
        加载单个视频的分辨率（用于并行预加载）。
        """
        if not os.path.exists(video_path):
            return None, (1920, 1080)

        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return os.path.basename(video_path), (width, height)
            cap.release()
        except Exception:
            pass
        return os.path.basename(video_path), (1920, 1080)

    @staticmethod
    def _preload_video_resolutions(ground_truths, video_dir=None, num_workers=4):
        """
        批量预加载所有视频的分辨率（并行）。

        Args:
            ground_truths: 真实标签列表
            video_dir: 视频目录
            num_workers: 并行线程数

        Returns:
            dict: {video_name: (width, height)} 映射
        """
        res_cache = {}
        video_tasks = []

        # 收集所有需要加载的视频路径
        for gt in ground_truths:
            current_video_dir = gt.get("_video_dir") or video_dir
            if current_video_dir:
                video_name = gt.get("video_name")
                if video_name and video_name not in res_cache:
                    video_path = os.path.join(current_video_dir, video_name)
                    video_tasks.append(video_path)

        # 去重
        video_tasks = list(set(video_tasks))

        if not video_tasks:
            return res_cache

        # 并行加载分辨率
        if num_workers > 1 and len(video_tasks) > 1:
            with ThreadPoolExecutor(max_workers=min(num_workers, len(video_tasks))) as executor:
                futures = {executor.submit(Evaluator._load_video_resolution, vp): vp
                           for vp in video_tasks}
                for future in as_completed(futures):
                    try:
                        video_name, resolution = future.result()
                        if video_name:
                            res_cache[video_name] = resolution
                    except Exception:
                        pass
        else:
            # 单线程模式
            for video_path in video_tasks:
                video_name, resolution = Evaluator._load_video_resolution(
                    video_path)
                if video_name:
                    res_cache[video_name] = resolution

        return res_cache

    @staticmethod
    def _evaluate_single_sample_wrapper(args):
        """
        评估单个样本的包装函数（用于多线程）。
        """
        pred, gt, width, height = args
        return Evaluator.evaluate_sample(pred, gt, width=width, height=height)

    @staticmethod
    def evaluate_batch(predictions, ground_truths, video_dir=None, num_workers=None):
        """
        批量评估结果（支持多线程并行）。

        Args:
            predictions: 预测结果列表
            ground_truths: 真实标签列表
            video_dir: 视频目录
            num_workers: 并行评估线程数，None 表示使用单线程
        """
        if not predictions or not ground_truths:
            return {}

        # 预加载所有视频分辨率（并行）
        res_cache = Evaluator._preload_video_resolutions(
            ground_truths, video_dir, num_workers=num_workers or 4)

        # 准备评估参数
        eval_args = []
        for pred, gt in zip(predictions, ground_truths):
            width, height = 1920, 1080  # 默认值

            video_name = gt.get("video_name")
            if video_name and video_name in res_cache:
                width, height = res_cache[video_name]

            eval_args.append((pred, gt, width, height))

        # 并行评估
        results_with_gt = []  # Store (score, gt) pairs to keep association
        if num_workers and num_workers > 1 and len(eval_args) > 1:
            with ThreadPoolExecutor(max_workers=min(num_workers, len(eval_args))) as executor:
                futures = {executor.submit(Evaluator._evaluate_single_sample_wrapper, args): args
                           for args in eval_args}
                for future in as_completed(futures):
                    # Retrieve the original args (specifically GT) associated with this future
                    _, gt, _, _ = futures[future]
                    try:
                        score = future.result()
                        results_with_gt.append((score, gt))
                    except Exception as e:
                        # 如果评估失败，添加默认分数
                        default_score = {
                            "intent_accuracy": 0.0,
                            "spatial_grounding": [],
                            "temporal_grounding": None
                        }
                        results_with_gt.append((default_score, gt))
        else:
            # 单线程模式
            for args in eval_args:
                _, gt, _, _ = args
                try:
                    score = Evaluator._evaluate_single_sample_wrapper(args)
                    results_with_gt.append((score, gt))
                except Exception as e:
                    default_score = {
                        "intent_accuracy": 0.0,
                        "spatial_grounding": [],
                        "temporal_grounding": None
                    }
                    results_with_gt.append((default_score, gt))

        if not results_with_gt:
            return {}

        # Separate scores for global calculation
        all_scores = [x[0] for x in results_with_gt]

        avg_intent = sum(s["intent_accuracy"]
                         for s in all_scores) / len(all_scores)

        # 空间和时间 Grounding 取平均
        all_spatial = []
        for s in all_scores:
            all_spatial.extend(s["spatial_grounding"])
        avg_spatial = sum(all_spatial) / \
            len(all_spatial) if all_spatial else 0.0

        all_temporal = []
        for s in all_scores:
            if s["temporal_grounding"] is not None:
                all_temporal.extend(s["temporal_grounding"])
        avg_temporal = sum(all_temporal) / \
            len(all_temporal) if all_temporal else 0.0

        # 按指令类型统计 (Instruction-wise breakdown)
        instruction_breakdown = {}
        # NOTE: We must use results_with_gt because all_scores (from as_completed) is not in sync with eval_args
        for score, gt in results_with_gt:
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
            ib["spatial_grounding"].extend(score["spatial_grounding"])
            if score["temporal_grounding"] is not None:
                ib["temporal_grounding"].extend(score["temporal_grounding"])

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
            "intent_grounding_accuracy": avg_intent,
            "spatial_grounding_accuracy": avg_spatial,
            "temporal_grounding_accuracy": avg_temporal,
            "overall_score": (avg_intent + avg_spatial + avg_temporal) / 3 if avg_temporal > 0 else (avg_intent + avg_spatial) / 2,
            "instruction_breakdown": breakdown_results
        }
