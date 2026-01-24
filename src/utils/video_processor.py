# src/utils/video_processor.py
import cv2
import os
import numpy as np
import logging
import subprocess
import tempfile
import base64
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("VSIG_Logger")


class VideoProcessor:
    """
    处理视频输入，提取帧用于模型推理。
    """

    @staticmethod
    def extract_frames(video_path, num_frames=8, end_timestamp_sec=None):
        """
        从视频中均匀提取多帧。

        Args:
            video_path (str): 视频文件路径。
            num_frames (int): 提取帧的数量。
            end_timestamp_sec (float, optional): 结束时间戳。如果提供，则在 0 到 end_timestamp_sec 之间抽样。

        Returns:
            list: 保存的临时图像路径列表。
            str: 最后一帧（用于 Grounding）的路径。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件未找到: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # 尝试使用 ffmpeg 作为备选方案
            logger.warning(f"OpenCV 无法打开视频: {video_path}，尝试使用 ffmpeg...")
            try:
                return VideoProcessor._extract_frames_with_ffmpeg(
                    video_path, num_frames, end_timestamp_sec)
            except Exception as ffmpeg_error:
                logger.error(f"ffmpeg 也失败: {ffmpeg_error}")
                raise ValueError(f"无法打开视频: {video_path} (OpenCV 和 ffmpeg 都失败)")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            cap.release()
            logger.warning(
                f"视频参数无效 (fps={fps}, frames={total_frames}): {video_path}，尝试使用 ffmpeg...")
            try:
                return VideoProcessor._extract_frames_with_ffmpeg(
                    video_path, num_frames, end_timestamp_sec)
            except Exception as ffmpeg_error:
                logger.error(f"ffmpeg 也失败: {ffmpeg_error}")
                raise ValueError(
                    f"视频参数无效: {video_path} (fps={fps}, frames={total_frames})")

        start_frame = 0
        if end_timestamp_sec is None:
            end_frame = total_frames - 1
        else:
            end_frame = min(int(end_timestamp_sec * fps), total_frames - 1)

        # 均匀采样帧索引
        if end_frame < num_frames:
            # 如果可用帧数少于需要的帧数，则取所有可用帧，并在后面重复最后一帧（或者就只取这么多）
            # 简单起见，这里我们取所有帧，并在列表中可能会少于 num_frames
            frame_indices = np.linspace(
                start_frame, end_frame, end_frame - start_frame + 1, dtype=int)
        else:
            frame_indices = np.linspace(
                start_frame, end_frame, num_frames, dtype=int)

        frame_paths = []
        last_frame_path = None

        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        video_name = os.path.basename(video_path).split('.')[0]

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                filename = f"{video_name}_frame_{idx}.jpg"
                save_path = os.path.join(temp_dir, filename)
                # 将 BGR 转换为 RGB，并使用 PIL 保存以保持正确的色彩空间和曝光度
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_pil.save(save_path, 'JPEG', quality=95)
                frame_paths.append(save_path)

        # 提取视频的最后一帧用于可视化（无论是否有 end_timestamp_sec）
        last_frame_idx = total_frames - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_idx)
        ret, last_frame = cap.read()
        if ret:
            filename = f"{video_name}_frame_{last_frame_idx}.jpg"
            last_frame_path = os.path.join(temp_dir, filename)
            # 将 BGR 转换为 RGB，并使用 PIL 保存以保持正确的色彩空间和曝光度
            last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(last_frame_rgb)
            img_pil.save(last_frame_path, 'JPEG', quality=95)
        elif frame_paths:
            # 如果无法读取最后一帧，使用采样帧中的最后一帧作为备选
            last_frame_path = frame_paths[-1]

        cap.release()

        if not frame_paths:
            # 如果 OpenCV 失败，尝试使用 ffmpeg
            logger.warning(f"OpenCV 未能提取到任何帧: {video_path}，尝试使用 ffmpeg...")
            try:
                return VideoProcessor._extract_frames_with_ffmpeg(
                    video_path, num_frames, end_timestamp_sec)
            except Exception as ffmpeg_error:
                logger.error(f"ffmpeg 也失败: {ffmpeg_error}")
                raise ValueError(f"未能提取到任何帧: {video_path}")

        logger.debug(f"已提取 {len(frame_paths)} 帧，可视化帧: {last_frame_path}")
        return frame_paths, last_frame_path

    @staticmethod
    def extract_frame(video_path, timestamp_sec=None):
        """
        从视频中提取指定时间点的帧。如果未指定时间，提取最后一帧。

        Args:
            video_path (str): 视频文件路径。
            timestamp_sec (float, optional): 提取帧的时间戳（秒）。如果为 None，提取最后一帧。

        Returns:
            numpy.ndarray: 提取的图像帧 (BGR格式)。
            str: 保存的临时图像路径。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件未找到: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # 尝试使用 ffmpeg 作为备选方案
            logger.warning(f"OpenCV 无法打开视频: {video_path}，尝试使用 ffmpeg...")
            try:
                frame_paths, last_path = VideoProcessor._extract_frames_with_ffmpeg(
                    video_path, num_frames=1, end_timestamp_sec=timestamp_sec)
                frame = cv2.imread(last_path)
                return frame, last_path
            except Exception as ffmpeg_error:
                logger.error(f"ffmpeg 也失败: {ffmpeg_error}")
                raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            cap.release()
            logger.warning(
                f"视频参数无效 (fps={fps}, frames={total_frames}): {video_path}，尝试使用 ffmpeg...")
            try:
                frame_paths, last_path = VideoProcessor._extract_frames_with_ffmpeg(
                    video_path, num_frames=1, end_timestamp_sec=timestamp_sec)
                frame = cv2.imread(last_path)
                return frame, last_path
            except Exception as ffmpeg_error:
                logger.error(f"ffmpeg 也失败: {ffmpeg_error}")
                raise ValueError(f"视频参数无效: {video_path}")

        if timestamp_sec is None:
            # 默认取最后一帧
            target_frame_idx = total_frames - 1
        else:
            target_frame_idx = int(timestamp_sec * fps)

        if target_frame_idx >= total_frames:
            target_frame_idx = total_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None or frame.size == 0:
            # 如果 OpenCV 读取失败，尝试使用 ffmpeg
            logger.warning(f"OpenCV 无法读取视频帧: {video_path}，尝试使用 ffmpeg...")
            try:
                frame_paths, last_path = VideoProcessor._extract_frames_with_ffmpeg(
                    video_path, num_frames=1, end_timestamp_sec=timestamp_sec)
                frame = cv2.imread(last_path)
                return frame, last_path
            except Exception as ffmpeg_error:
                logger.error(f"ffmpeg 也失败: {ffmpeg_error}")
                raise ValueError("无法读取视频帧")

        # 保存临时文件用于模型输入
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        filename = os.path.basename(video_path).split(
            '.')[0] + f"_frame_{target_frame_idx}.jpg"
        save_path = os.path.join(temp_dir, filename)
        # 将 BGR 转换为 RGB，并使用 PIL 保存以保持正确的色彩空间和曝光度
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_pil.save(save_path, 'JPEG', quality=95)

        logger.debug(f"已提取帧: {save_path} (Index: {target_frame_idx})")
        return frame, save_path

    @staticmethod
    def visualize_points(image_path, result_json, output_path, gt_json=None, gt_items=None):
        """
        在图像上绘制预测点，用于可视化结果。

        Args:
            image_path (str): 原始图像路径。
            result_json (dict): 模型输出的 JSON 结果。
            output_path (str): 保存可视化结果的路径。
            gt_json (dict, optional): 原始 Ground Truth 数据，用于对比（作为备选）。
            gt_items (list, optional): 处理后的 GT items 列表 (包含合并/筛选逻辑)。
        """
        if not os.path.exists(image_path):
            logger.warning(
                f"Warning: Image {image_path} not found for visualization.")
            return

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        # 解析坐标
        # 预测点与 GT 点此时均已统一为 [x, y] 格式
        pred_point_list = result_json.get("point_list", [])

        # GT 点：优先使用 gt_items（已处理，points为像素坐标），否则回退到 gt_json["object_space"]
        gt_point_list = []
        gt_cmd = ""

        # 必须提供 gt_items，否则无法可视化
        if gt_items is None or len(gt_items) == 0:
            raise ValueError(
                "无法可视化：缺少 GT items 数据。"
                "请确保从 eval_gt.json 正确加载了评估数据。"
            )
        
        # gt_items 已经过处理，points 已经是像素坐标 [x, y]
        source_items = gt_items
        logger.debug(f"使用 gt_items，共 {len(source_items)} 个GT项")

        # 从 gt_json 中获取 choices 信息（用于从 choice 字段获取名称）
        object_choices = []
        space_choices = []
        if gt_json:
            object_choices = gt_json.get("_object_choices", [])
            space_choices = gt_json.get("_space_choices", [])
        
        # 构建 choice 到名称的映射
        choice_to_name = {}
        for choice_str in object_choices + space_choices:
            # 格式: "A. 名称" 或 "a. 名称"
            if ". " in choice_str:
                parts = choice_str.split(". ", 1)
                if len(parts) == 2:
                    choice_to_name[parts[0]] = parts[1]

        # 提取GT信息
        for point in source_items:
            # 尝试提取 mask 数据
            mask_data = None
            if "mask" in point:
                mask_data = point["mask"]

            # 优先使用 name 字段，如果没有则从 choice 字段获取
            point_name = point.get("name", "")
            if not point_name:
                choice = point.get("choice", "")
                if choice and choice in choice_to_name:
                    point_name = choice_to_name[choice]
                elif choice:
                    # 如果找不到对应的名称，至少显示 choice
                    point_name = f"选项 {choice}"
            
            if point_name:
                gt_cmd = gt_cmd + point_name + " -> "

            points = point.get("points")
            # 即使没有points，如果有mask或points，也要添加到列表中
            if points or mask_data:
                gt_point_list.append({
                    "type": point.get("type"),
                    "description": point_name,
                    "point": points if points else [],  # 已经是像素坐标 [x, y] 或 [[x, y], ...]
                    "mask": mask_data
                })
                logger.debug(f"添加GT项: name={point_name}, type={point.get('type')}, has_points={points is not None}, has_mask={mask_data is not None}")

        if len(gt_point_list) == 0:
            logger.warning(f"GT点列表为空，将只显示预测结果")

        def norm_to_pixel(pt):
            """
            将归一化坐标 [x, y] (范围 0-1000) 转换为像素坐标 (x_pixel, y_pixel)
            支持单个点 [x, y] 或多个点 [[x1, y1], [x2, y2], ...] 格式
            Args:
                pt: [x, y] 或 [[x1, y1], [x2, y2], ...] 格式，x 是水平方向，y 是垂直方向
            Returns:
                单个点返回 (x_pixel, y_pixel) 元组，多个点返回 [(x1, y1), (x2, y2), ...] 列表
            """
            if not pt:
                return None

            # 检查是否是多个点的列表格式 [[x1, y1], [x2, y2], ...]
            if isinstance(pt[0], list):
                result = []
                for point in pt:
                    if len(point) >= 2:
                        x_norm = point[0]  # x 是水平方向，第一个元素
                        y_norm = point[1]  # y 是垂直方向，第二个元素
                        x_pixel = int(x_norm / 1000 * w)
                        y_pixel = int(y_norm / 1000 * h)
                        result.append((x_pixel, y_pixel))
                return result if result else None

            # 单个点格式 [x, y]
            if len(pt) < 2:
                return None
            x_norm = pt[0]  # x 是水平方向，第一个元素
            y_norm = pt[1]  # y 是垂直方向，第二个元素
            x_pixel = int(x_norm / 1000 * w)
            y_pixel = int(y_norm / 1000 * h)
            return (x_pixel, y_pixel)

        # 定义类型对应的颜色
        # target_object: Red (0, 0, 255)
        # spatial_affordance: Green (0, 255, 0)
        # object: Yellow (0, 255, 255) for GT
        # space: Cyan (255, 255, 0) for GT
        def get_color_for_type(point_type, is_pred=True):
            """根据类型返回颜色"""
            if is_pred:
                if point_type == "target_object":
                    return (0, 0, 255)  # Red
                elif point_type == "spatial_affordance":
                    return (0, 255, 0)  # Green
            else:
                if point_type == "object":
                    return (0, 255, 255)  # Yellow
                elif point_type == "space":
                    return (255, 255, 0)  # Cyan
            return (255, 255, 255)  # White as default

        # 1. 绘制 GT 点/Mask
        # 创建 Mask 叠加层
        overlay = img.copy()

        for idx, point_item in enumerate(gt_point_list):
            if not isinstance(point_item, dict):
                continue

            point_type = point_item.get("type", "")
            point_coord = point_item.get("point", [])
            mask_info = point_item.get("mask")
            description = point_item.get("description", "")

            color = get_color_for_type(point_type, is_pred=False)

            # 绘制 Mask（如果有）
            has_drawn_mask = False
            if mask_info and "mask_base64" in mask_info:
                try:
                    # 解码 Mask
                    mask_data = base64.b64decode(mask_info["mask_base64"])
                    nparr = np.frombuffer(mask_data, np.uint8)
                    mask_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

                    if mask_img is not None:
                        bbox = mask_info.get("bbox")  # [x1, y1, x2, y2]
                        if bbox:
                            # 修正坐标顺序: [x1, y1, x2, y2]
                            bx1, by1, bx2, by2 = bbox
                            bx1, by1 = max(0, int(bx1)), max(0, int(by1))
                            bx2, by2 = min(w, int(bx2)), min(h, int(by2))

                            if mask_img.shape[0] == h and mask_img.shape[1] == w:
                                resized_mask = mask_img
                            else:
                                target_h, target_w = by2 - by1, bx2 - bx1
                                if target_h > 0 and target_w > 0:
                                    resized_mask_patch = cv2.resize(
                                        mask_img, (target_w, target_h))
                                    resized_mask = np.zeros(
                                        (h, w), dtype=np.uint8)
                                    resized_mask[by1:by2,
                                                 bx1:bx2] = resized_mask_patch
                                else:
                                    resized_mask = None
                        else:
                            resized_mask = cv2.resize(mask_img, (w, h)) if (
                                mask_img.shape[0] != h or mask_img.shape[1] != w) else mask_img

                        if resized_mask is not None:
                            contours, _ = cv2.findContours(
                                resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(
                                overlay, contours, -1, color, -1)
                            has_drawn_mask = True
                except Exception as e:
                    logger.error(f"绘制 Mask 失败: {e}")

            # 绘制 GT 点（总是绘制，即使有mask也绘制点以便更清晰）
            if point_coord:
                if isinstance(point_coord[0], list):
                    # 多个点 [[x, y], ...] - GT points 已经是像素坐标
                    for pt in point_coord:
                        if len(pt) >= 2:
                            pixel_pt = (int(pt[0]), int(pt[1]))
                            cv2.circle(img, pixel_pt, 10, color, 3)
                            cv2.circle(img, pixel_pt, 3, color, -1)
                            # 绘制GT标签
                            cv2.putText(img, "GT", (pixel_pt[0]+12, pixel_pt[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    # 单个点 [x, y] - GT points 已经是像素坐标
                    if len(point_coord) >= 2:
                        pixel_pt = (int(point_coord[0]), int(point_coord[1]))
                        cv2.circle(img, pixel_pt, 10, color, 3)
                        cv2.circle(img, pixel_pt, 3, color, -1)
                        # 绘制GT标签
                        cv2.putText(img, "GT", (pixel_pt[0]+12, pixel_pt[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 混合图层 (Mask 透明度)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # 2. 按顺序绘制预测点 (绘制在混合层之上，确保清晰)
        for idx, point_item in enumerate(pred_point_list):
            if not isinstance(point_item, dict):
                continue

            point_type = point_item.get("type", "")
            point_coord = point_item.get("point", [])

            if not point_coord:
                continue

            color = get_color_for_type(point_type, is_pred=True)
            if isinstance(point_coord[0], list):
                # 多个点的情况 [[x1, y1], [x2, y2]]
                for i, pt in enumerate(point_coord):
                    x, y = pt[0], pt[1]
                    # 预测点通常是归一化坐标 0-1000
                    pixel_pt = (int(x / 1000 * w), int(y / 1000 * h))
                    cv2.circle(img, pixel_pt, 10, color, -1)
                    if i == 0:
                        cv2.putText(img, "Pred", (pixel_pt[0]+10, pixel_pt[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # 单个点的情况 [x, y]
                x, y = point_coord[0], point_coord[1]
                pixel_pt = (int(x / 1000 * w), int(y / 1000 * h))
                cv2.circle(img, pixel_pt, 10, color, -1)
                cv2.putText(img, "Pred", (pixel_pt[0]+10, pixel_pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. 绘制指令文本和标签
        pred_cmd = result_json.get("explicit_command", "")

        # 使用 Pillow 绘制中文
        try:
            # OpenCV图片转PIL图片
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # 获取项目根目录（假设当前文件在 src/utils/ 目录下）
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_file_dir))
            project_font_path = os.path.join(project_root, "fonts", "SourceHanSansCN-Regular.otf")

            # 尝试加载中文字体（优先使用项目文件夹中的字体）
            font_paths = [
                project_font_path,                           # 项目字体（优先）
                "/System/Library/Fonts/STHeiti Light.ttc",  # macOS
                "/System/Library/Fonts/PingFang.ttc",       # macOS
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
                "C:\\Windows\\Fonts\\msyh.ttc",             # Windows
                "simhei.ttf"                                # Common fallback
            ]

            font = None
            for path in font_paths:
                if os.path.exists(path):
                    try:
                        font = ImageFont.truetype(path, 30)  # 字号 30
                        logger.debug(f"成功加载字体: {path}")
                        break
                    except Exception as e:
                        logger.debug(f"加载字体失败 {path}: {e}")
                        continue

            if font is None:
                logger.warning("未能加载任何中文字体，使用默认字体（可能无法正确显示中文）")
                font = ImageFont.load_default()

            # 绘制文字
            # Pred Cmd
            draw.text((20, 40), f"Cmd (Pred): {pred_cmd}", font=font, fill=(
                255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))

            # GT Cmd (恢复显示)
            if gt_cmd:
                draw.text((20, 80), f"Cmd (GT): {gt_cmd}", font=font, fill=(
                    0, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))

            # 绘制 GT 标签 (直接显示物体名称)
            for idx, point_item in enumerate(gt_point_list):
                description = point_item.get("description", "")
                point_type = point_item.get("type", "")

                if not description:
                    continue

                # 确定标签位置 - GT points 已经是像素坐标
                label_pos = None
                point_coord = point_item.get("point", [])
                if point_coord:
                    if isinstance(point_coord[0], list):
                        pt = point_coord[0]  # 使用第一个点作为标签位置
                    else:
                        pt = point_coord

                    if len(pt) >= 2:
                        label_pos = (int(pt[0]), int(pt[1]))

                if label_pos:
                    # 使用 get_color_for_type 统一颜色逻辑 (Pillow 使用 RGB)
                    bgr_color = get_color_for_type(point_type, is_pred=False)
                    rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])

                    # 绘制GT名称标签，位置在点的右上方
                    draw.text((label_pos[0] + 15, label_pos[1] - 25), 
                              f"GT: {description}",
                              font=font, fill=rgb_color, stroke_width=2, stroke_fill=(0, 0, 0))

            # PIL图片转回OpenCV图片
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.error(f"绘制中文文本失败: {e}")
            # 降级回 OpenCV 绘制 (可能会乱码)
            cv2.putText(img, f"Cmd (Pred): {pred_cmd[:30]}...", (
                20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if gt_cmd:
                cv2.putText(img, f"Cmd (GT): {gt_cmd[:30]}...", (
                    20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 降级时也尝试绘制 GT 标签名
            for idx, point_item in enumerate(gt_point_list):
                description = point_item.get("description", "")
                point_type = point_item.get("type", "")
                point_coord = point_item.get("point", [])
                if point_coord and description:
                    if isinstance(point_coord[0], list): 
                        pt = point_coord[0]
                    else: 
                        pt = point_coord
                    if len(pt) >= 2:
                        color = get_color_for_type(point_type, is_pred=False)
                        cv2.putText(img, f"GT: {description[:15]}", (int(pt[0])+15, int(pt[1])-25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imwrite(output_path, img)
        logger.info(f"可视化结果已保存: {output_path}")

    @staticmethod
    def _extract_frames_with_ffmpeg(video_path, num_frames=8, end_timestamp_sec=None):
        """
        使用 ffmpeg 提取视频帧（备选方案，当 OpenCV 失败时使用）。
        """
        try:
            # 获取视频时长
            probe_cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path
            ]
            result = subprocess.run(
                probe_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise ValueError(f"无法获取视频时长: {result.stderr}")

            duration = float(result.stdout.strip())
            if end_timestamp_sec is not None:
                duration = min(duration, end_timestamp_sec)

            # 安全阈值：避开视频最后 0.2s 的可能损坏区
            safe_duration = max(0, duration - 0.2)

            # 计算采样时间点
            if num_frames > 1:
                time_points = np.linspace(0, safe_duration, num_frames)
            else:
                time_points = [safe_duration]

            frame_paths = []
            temp_dir = "temp_frames"
            os.makedirs(temp_dir, exist_ok=True)
            video_name = os.path.basename(video_path).split('.')[0]

            for i, t in enumerate(time_points):
                output_file = os.path.join(
                    temp_dir, f"{video_name}_frame_ffmpeg_{i}.jpg")

                # 策略 1: 快速 Seek + 强制兼容像素格式
                cmd = [
                    "ffmpeg", "-y", "-ss", str(t), "-i", video_path,
                    "-vframes", "1", "-q:v", "2",
                    "-pix_fmt", "yuvj420p",  # 强制 MJPEG 兼容的像素格式
                    "-strict", "-2",         # 允许处理非标准转换
                    "-f", "image2", output_file
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30)

                # 如果策略 1 失败，且是最后一帧，尝试模仿标注工具的末尾寻帧逻辑 (-sseof)
                if (result.returncode != 0 or not os.path.exists(output_file) or os.path.getsize(output_file) == 0) and i == len(time_points) - 1:
                    logger.warning(f"ffmpeg 寻帧失败，尝试标注工具的 -sseof 兜底模式...")
                    cmd = [
                        "ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path,
                        "-vframes", "1", "-q:v", "2",
                        "-pix_fmt", "yuvj420p", "-f", "image2", output_file
                    ]
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=30)

                # 如果依然失败，尝试策略 2: 精确 Seek
                if result.returncode != 0 or not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                    logger.warning(f"ffmpeg 快速提取失败 (t={t}s)，尝试精确提取模式...")
                    cmd = [
                        "ffmpeg", "-y", "-i", video_path, "-ss", str(t),
                        "-vframes", "1", "-q:v", "2",
                        "-pix_fmt", "yuvj420p",
                        "-strict", "-2",
                        "-f", "image2", output_file
                    ]
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=30)

                if result.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    frame_paths.append(output_file)
                else:
                    # 最后的兜底：如果所有模式都失败，尝试直接提取第一帧，保证流程不中断
                    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", video_path, "-vframes", "1", output_file], capture_output=True)
                        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                            frame_paths.append(output_file)

                    if output_file not in frame_paths:
                        logger.warning(
                            f"ffmpeg 所有提取模式均失败 (t={t}s): {result.stderr}")

            if not frame_paths:
                raise ValueError("ffmpeg 未能提取到任何帧")

            # 最后一帧用于可视化
            last_frame_path = frame_paths[-1]

            logger.info(f"使用 ffmpeg 提取了 {len(frame_paths)} 帧")
            return frame_paths, last_frame_path

        except FileNotFoundError:
            raise ValueError("ffmpeg 未安装，无法使用备选方案")
        except Exception as e:
            raise ValueError(f"ffmpeg 提取失败: {e}")
