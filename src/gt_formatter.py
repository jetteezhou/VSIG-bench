"""
GT格式化模块：从annotation和description中提取评估所需的GT格式
统一预处理所有后续逻辑需要的信息：视频分辨率、最后一帧、坐标转换等
"""
import os
import json
import cv2
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.video_processor import VideoProcessor


class GTFormatter:
    """GT格式化器：统一处理GT数据的格式化和提取"""

    # GT 处理缓存
    _gt_cache = {}

    @staticmethod
    def _convert_gt_points_to_xy(raw_os: List[Dict]) -> List[Dict]:
        """
        将 GT points 从 [y, x] 格式转换为 [x, y] 格式。

        重要：此转换在数据加载的最早阶段完成，确保后续所有逻辑都使用 [x, y] 格式。
        不暴露 [y, x] 到外部逻辑。

        Args:
            raw_os: 原始 object_space 列表，points 为 [y, x] 格式

        Returns:
            转换后的列表，points 为 [x, y] 格式
        """
        processed_raw_os = []
        for item in raw_os:
            new_item = item.copy()
            points = item.get("points", [])
            if points:
                if isinstance(points[0], list):
                    # [[y1, x1], [y2, x2]] -> [[x1, y1], [x2, y2]]
                    new_item["points"] = [[p[1], p[0]] for p in points]
                else:
                    # [y, x] -> [x, y]
                    new_item["points"] = [points[1], points[0]]
            processed_raw_os.append(new_item)
        return processed_raw_os

    @staticmethod
    def _normalize_to_pixel_coords(points, width: int, height: int):
        """
        将归一化坐标（0-1000）转换为像素坐标。

        Args:
            points: 归一化坐标 [x, y] 或 [[x, y], ...]
            width: 视频宽度
            height: 视频高度

        Returns:
            像素坐标 [x, y] 或 [[x, y], ...]
        """
        if not points:
            return points

        if isinstance(points[0], list):
            # 多个点
            return [[int(p[0] * width / 1000), int(p[1] * height / 1000)] for p in points]
        else:
            # 单个点
            return [int(points[0] * width / 1000), int(points[1] * height / 1000)]

    @staticmethod
    def _extract_video_info(video_path: str, extract_last_frame: bool = False) -> Dict:
        """
        提取视频信息：分辨率、最后一帧路径等。

        Args:
            video_path: 视频文件路径
            extract_last_frame: 是否提取最后一帧（默认False，因为很慢，延迟到需要时再提取）

        Returns:
            包含 width, height, last_frame_path 的字典
        """
        info = {
            "width": 1920,
            "height": 1080,
            "last_frame_path": None
        }

        if not os.path.exists(video_path):
            return info

        try:
            # 提取视频分辨率（快速操作）
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

            # 提取最后一帧（慢操作，默认不提取，延迟到需要时）
            if extract_last_frame:
                try:
                    _, last_frame_path = VideoProcessor.extract_frame(
                        video_path, timestamp_sec=None)
                    info["last_frame_path"] = last_frame_path
                except Exception:
                    pass  # 如果提取失败，保持 None
        except Exception:
            pass  # 如果出错，使用默认值

        return info

    @staticmethod
    def parse_description_file(file_path: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        解析 description.txt 文件，提取选项定义和答案。

        Args:
            file_path: description.txt文件路径

        Returns:
            (选项定义文本, 视频答案映射)
            - definitions (str): 格式化的选项定义字符串，用于 Prompt。
            - video_answers (dict): 视频文件名到正确选项的映射 {video_name: [options]}.
        """
        if not os.path.exists(file_path):
            return "", {}

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        definitions = []
        video_answers = {}

        # 简单的状态机解析
        # 1. 解析定义部分（第一步）
        # 2. 解析答案部分（第二步）

        part1_lines = []
        part2_lines = []

        current_part = 0
        for line in lines:
            if "第一步" in line:
                current_part = 1
                continue
            elif "第二步" in line:
                current_part = 2
                continue

            if current_part == 1:
                if line.strip() and not line.startswith("-") and not line.startswith("="):
                    part1_lines.append(line)
            elif current_part == 2:
                if line.strip() and not line.startswith("-") and not line.startswith("="):
                    part2_lines.append(line)

        # 格式化定义部分
        definitions = "\n".join(part1_lines).strip()

        # 解析答案部分
        # 格式: VIDxxxx.mp4: A, B
        for line in part2_lines:
            if ":" in line:
                parts = line.split(":")
                video_name = parts[0].strip()
                answers_str = parts[1].strip()
                # 处理可能的逗号分隔（支持中文和英文逗号）
                answers = [ans.strip() for ans in answers_str.replace(
                    "，", ",").split(",") if ans.strip()]
                video_answers[video_name] = answers

        return definitions, video_answers

    @staticmethod
    def process_gt_by_template(gt: Dict) -> Dict:
        """
        根据任务模板对 GT 数据进行预处理，特别是指令 4/5/6 的合并逻辑。
        使用缓存优化重复处理。

        重要：GT points 坐标转换
        - 输入：annotation 中的 points 为 [y, x] 格式
        - 输出：统一转换为 [x, y] 格式，供后续所有逻辑使用
        - 此转换在数据加载的最早阶段完成，不暴露 [y, x] 到外部逻辑

        Args:
            gt: 单个annotation项，包含 task_template 和 object_space

        Returns:
            处理后的GT数据，包含：
            - template: 任务模板
            - items: 处理后的GT items列表（points 已转换为 [x, y] 格式）
        """
        # 使用 GT 的 JSON 字符串作为缓存键
        gt_key = json.dumps(gt, sort_keys=True, ensure_ascii=False)

        if gt_key in GTFormatter._gt_cache:
            return GTFormatter._gt_cache[gt_key]

        template = gt.get("task_template")
        raw_os = gt.get("object_space", [])

        # GT points 坐标转换：将标注中的 [y, x] 统一转换为全系统通用的 [x, y]
        raw_os = GTFormatter._convert_gt_points_to_xy(raw_os)

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

        GTFormatter._gt_cache[gt_key] = processed_gt
        return processed_gt

    @staticmethod
    def format_gt_for_evaluation(
        annotation_item: Dict,
        video_dir: str,
        answers_map: Dict[str, List[str]] = None,
        video_width: int = None,
        video_height: int = None
    ) -> Dict:
        """
        格式化GT数据用于评估，统一预处理所有后续逻辑需要的信息。

        Args:
            annotation_item: 单个annotation项（points 为 [y, x] 格式）
            video_dir: 视频目录路径
            answers_map: 从description.txt解析出的答案映射 {video_name: [options]}
            video_width: 视频宽度（如果已提供，避免重复提取）
            video_height: 视频高度（如果已提供，避免重复提取）

        Returns:
            格式化后的GT数据，包含：
            - 原始annotation数据
            - _video_dir: 视频目录路径
            - _video_width, _video_height: 视频分辨率（像素）
            - _last_frame_path: 最后一帧路径（用于可视化）
            - _processed_gt: 处理后的GT items（points 已转换为像素坐标 [x, y]）
            - _correct_options: 正确答案选项列表（用于intent评估）

        重要：统一预处理
        - GT points: [y, x] -> [x, y] -> 像素坐标
        - 视频信息：分辨率、最后一帧路径
        - 所有坐标统一为像素坐标，后续逻辑无需转换
        """
        # 确保有_video_dir字段
        formatted_gt = annotation_item.copy()
        formatted_gt["_video_dir"] = video_dir

        # 提取视频信息（如果未提供分辨率，则提取）
        video_name = annotation_item.get("video_name")
        video_path = os.path.join(
            video_dir, video_name) if video_name else None

        if video_width is not None and video_height is not None:
            # 使用提供的分辨率（批量提取时已缓存）
            width, height = video_width, video_height
        elif video_path:
            # 单独提取分辨率（慢，仅在单独调用时使用）
            video_info = GTFormatter._extract_video_info(
                video_path, extract_last_frame=False)
            width, height = video_info["width"], video_info["height"]
        else:
            width, height = 1920, 1080

        formatted_gt["_video_width"] = width
        formatted_gt["_video_height"] = height
        formatted_gt["_video_path"] = video_path  # 保存视频路径，需要时再提取最后一帧
        formatted_gt["_last_frame_path"] = None  # 延迟提取

        # 处理GT items（用于空间/时间评估）
        # process_gt_by_template 会将 points 从 [y, x] 转换为 [x, y]（归一化坐标 0-1000）
        processed_gt = GTFormatter.process_gt_by_template(formatted_gt)

        # 将GT points从归一化坐标转换为像素坐标（使用已确定的 width 和 height）
        for item in processed_gt.get("items", []):
            if "points" in item:
                item["points"] = GTFormatter._normalize_to_pixel_coords(
                    item["points"], width, height
                )

        formatted_gt["_processed_gt"] = processed_gt

        # 提取正确答案选项（用于intent评估）
        correct_options = []
        if answers_map and video_name in answers_map:
            correct_options = answers_map[video_name]
        formatted_gt["_correct_options"] = correct_options

        return formatted_gt

    @staticmethod
    def _extract_video_resolution_batch(video_paths: List[str], num_workers: int = 4) -> Dict[str, Tuple[int, int]]:
        """
        批量并行提取视频分辨率。

        Args:
            video_paths: 视频路径列表
            num_workers: 并行线程数

        Returns:
            {video_path: (width, height)} 映射
        """
        resolution_cache = {}

        if not video_paths:
            return resolution_cache

        def extract_single(video_path):
            if not os.path.exists(video_path):
                return video_path, (1920, 1080)
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    return video_path, (width, height)
                cap.release()
            except Exception:
                pass
            return video_path, (1920, 1080)

        # 并行提取分辨率
        if num_workers > 1 and len(video_paths) > 1:
            with ThreadPoolExecutor(max_workers=min(num_workers, len(video_paths))) as executor:
                futures = {executor.submit(
                    extract_single, vp): vp for vp in video_paths}
                for future in as_completed(futures):
                    try:
                        video_path, resolution = future.result()
                        resolution_cache[video_path] = resolution
                    except Exception:
                        video_path = futures[future]
                        resolution_cache[video_path] = (1920, 1080)
        else:
            # 单线程模式
            for video_path in video_paths:
                _, resolution = extract_single(video_path)
                resolution_cache[video_path] = resolution

        return resolution_cache

    @staticmethod
    def format_batch_gt_for_evaluation(
        annotations: List[Dict],
        video_dir: str,
        answers_map: Dict[str, List[str]] = None,
        num_workers: int = 4
    ) -> List[Dict]:
        """
        批量格式化GT数据用于评估（并行提取视频分辨率以加快速度）

        Args:
            annotations: annotation数据列表
            video_dir: 视频目录路径
            answers_map: 从description.txt解析出的答案映射
            num_workers: 并行提取视频分辨率的线程数

        Returns:
            格式化后的GT数据列表
        """
        # 1. 批量并行提取所有视频的分辨率
        video_paths = []
        video_name_to_path = {}
        for item in annotations:
            video_name = item.get("video_name")
            if video_name:
                video_path = os.path.join(video_dir, video_name)
                video_paths.append(video_path)
                video_name_to_path[video_name] = video_path

        resolution_cache = GTFormatter._extract_video_resolution_batch(
            list(set(video_paths)), num_workers=num_workers
        )

        # 2. 格式化每个样本（使用缓存的分辨率）
        formatted_list = []
        for item in annotations:
            video_name = item.get("video_name")

            # 如果 answers_map 不为空，说明 description.txt 存在且被解析了
            # 如果视频不在 answers_map 中，说明在 generate_description.py 中被跳过了
            # 这种情况下应该跳过该视频，不进行后续评估
            if answers_map and len(answers_map) > 0:
                if video_name not in answers_map:
                    continue  # 跳过该视频

            video_path = video_name_to_path.get(
                video_name) if video_name else None

            # 从缓存中获取分辨率
            if video_path and video_path in resolution_cache:
                width, height = resolution_cache[video_path]
            else:
                width, height = 1920, 1080

            formatted_gt = GTFormatter.format_gt_for_evaluation(
                item, video_dir, answers_map, video_width=width, video_height=height
            )
            formatted_list.append(formatted_gt)

        return formatted_list
