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
        # 预测点：从 point_list 中按顺序提取，并将 [y, x] 转换为 [x, y]
        pred_point_list = result_json.get("point_list", [])

        # GT 点：优先使用 gt_items，否则回退到 gt_json["object_space"]
        gt_point_list = []
        gt_cmd = ""

        source_items = []
        if gt_items is not None:
            source_items = gt_items
        elif gt_json and "object_space" in gt_json and isinstance(gt_json.get("object_space"), list):
            source_items = gt_json.get("object_space")

        for point in source_items:
            # 尝试提取 mask 数据
            mask_data = None
            if "mask" in point:
                mask_data = point["mask"]

            gt_point_list.append({
                "type": point.get("type"),
                "description": point.get("name"),
                "point": point.get("points"),
                "mask": mask_data
            })
            gt_cmd = gt_cmd + point.get("name") + " -> "

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

        # 按顺序绘制预测点
        for idx, point_item in enumerate(pred_point_list):
            if not isinstance(point_item, dict):
                continue

            point_type = point_item.get("type", "")
            point_coord = point_item.get("point", [])

            if not point_coord:
                continue

            # 将 [y, x] 转换为 [x, y]
            if isinstance(point_coord, list) and len(point_coord) >= 2:
                if isinstance(point_coord[0], list):
                    # 多个点的情况
                    for i, pt in enumerate(point_coord):
                        x, y = pt[1], pt[0]  # Swap [y, x] to [x, y]
                        x_pixel = int(x / 1000 * w)
                        y_pixel = int(y / 1000 * h)
                        pixel_pt = (x_pixel, y_pixel)
                        color = get_color_for_type(point_type, is_pred=True)
                        cv2.circle(img, pixel_pt, 10, color, -1)
                        if i == 0:
                            label = "Pred"  # 简单标注
                            cv2.putText(img, label, (pixel_pt[0]+10, pixel_pt[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # 单个点的情况
                    # Swap [y, x] to [x, y]
                    x, y = point_coord[1], point_coord[0]
                    x_pixel = int(x / 1000 * w)
                    y_pixel = int(y / 1000 * h)
                    pixel_pt = (x_pixel, y_pixel)
                    color = get_color_for_type(point_type, is_pred=True)
                    cv2.circle(img, pixel_pt, 10, color, -1)
                    label = "Pred"
                    cv2.putText(img, label, (pixel_pt[0]+10, pixel_pt[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 按顺序绘制 GT 点/Mask
        # 创建 Mask 叠加层
        overlay = img.copy()

        for idx, point_item in enumerate(gt_point_list):
            if not isinstance(point_item, dict):
                continue

            point_type = point_item.get("type", "")
            point_coord = point_item.get("point", [])
            description = point_item.get("description", "")
            mask_info = point_item.get("mask")

            color = get_color_for_type(point_type, is_pred=False)

            # 优先绘制 Mask
            has_drawn_mask = False
            if mask_info and "mask_base64" in mask_info:
                try:
                    # 解码 Mask
                    mask_data = base64.b64decode(mask_info["mask_base64"])
                    nparr = np.frombuffer(mask_data, np.uint8)
                    mask_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

                    if mask_img is not None:
                        # 调整 mask 大小以匹配图像 (如果需要)
                        # 注意：mask 通常是全图大小或者 bbox 大小
                        # 这里假设 mask 对应的是全图或者 bbox，需要结合 bbox 使用
                        bbox = mask_info.get("bbox")  # [y1, x1, y2, x2]

                        if bbox:
                            x1, y1, x2, y2 = bbox
                            # 确保坐标有效
                            y1, x1, y2, x2 = max(0, y1), max(
                                0, x1), min(h, y2), min(w, x2)

                            # 如果 mask 是全图大小，直接 resize
                            if mask_img.shape[0] == h and mask_img.shape[1] == w:
                                resized_mask = mask_img
                            else:
                                # 如果 mask 是局部切片，resize 到 bbox 大小
                                target_h, target_w = y2 - y1, x2 - x1
                                if target_h > 0 and target_w > 0:
                                    resized_mask_patch = cv2.resize(
                                        mask_img, (target_w, target_h))
                                    # 创建全图大小的 mask
                                    resized_mask = np.zeros(
                                        (h, w), dtype=np.uint8)
                                    resized_mask[y1:y2,
                                                 x1:x2] = resized_mask_patch
                                else:
                                    resized_mask = None
                        else:
                            # 假设 mask 是全图
                            if mask_img.shape[0] != h or mask_img.shape[1] != w:
                                resized_mask = cv2.resize(mask_img, (w, h))
                            else:
                                resized_mask = mask_img

                        if resized_mask is not None:
                            # 找到轮廓并填充颜色
                            contours, _ = cv2.findContours(
                                resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(
                                overlay, contours, -1, color, -1)  # 填充
                            has_drawn_mask = True

                            # 在 Mask 中心标记文字
                            M = cv2.moments(contours[0]) if contours else None
                            if M and M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                # 稍后统一绘制中文文字
                                pass
                except Exception as e:
                    logger.error(f"绘制 Mask 失败: {e}")

            # 如果没有 Mask 或者绘制失败，回退到绘制点
            if not has_drawn_mask and point_coord:
                # GT 点是 [y, x] 格式，需要转换为 [x, y]
                converted_points = []
                if isinstance(point_coord[0], list):
                    for pt in point_coord:
                        if len(pt) >= 2:
                            x, y = pt[1], pt[0]
                            converted_points.append([x, y])
                elif len(point_coord) == 2:
                    x, y = point_coord[1], point_coord[0]
                    converted_points = [x, y]

                if converted_points:
                    pts = norm_to_pixel(converted_points)
                    if pts:
                        if isinstance(pts, list):
                            for pt in pts:
                                cv2.circle(img, pt, 8, color, 2)
                                cv2.circle(img, pt, 2, color, -1)
                        else:
                            cv2.circle(img, pts, 8, color, 2)
                            cv2.circle(img, pts, 2, color, -1)

        # 混合图层 (Mask 透明度)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # 绘制指令文本和标签 (最后绘制以防被覆盖)
        pred_cmd = result_json.get("explicit_command", "")

        # 使用 Pillow 绘制中文
        try:
            # OpenCV图片转PIL图片
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # 尝试加载中文字体
            font_paths = [
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
                        break
                    except Exception:
                        continue

            if font is None:
                font = ImageFont.load_default()

            # 绘制文字
            # Pred Cmd
            draw.text((20, 40), f"Cmd (Pred): {pred_cmd}", font=font, fill=(
                255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))

            # GT Cmd (恢复显示)
            if gt_cmd:
                draw.text((20, 80), f"Cmd (GT): {gt_cmd}", font=font, fill=(
                    0, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))

            # 2. 绘制 GT 标签 (直接显示物体名称)
            for idx, point_item in enumerate(gt_point_list):
                description = point_item.get("description", "")
                point_type = point_item.get("type", "")

                # 确定标签位置
                label_pos = None

                # 如果有 Mask，尝试用 Mask 中心
                if point_item.get("mask") and "mask_base64" in point_item["mask"]:
                    # (这里简单化，不重新解码计算重心，而是使用点位作为备选，或者在上面解码时保存了重心)
                    pass

                # 使用点位作为标签位置
                point_coord = point_item.get("point", [])
                if point_coord:
                    # [y, x] -> [x, y]
                    if isinstance(point_coord[0], list):
                        pt = point_coord[0]
                    else:
                        pt = point_coord

                    if len(pt) >= 2:
                        y, x = pt[0], pt[1]  # raw is [y, x] for GT
                        # Convert to pixel
                        x_pixel = int(x / 1000 * w)
                        y_pixel = int(y / 1000 * h)
                        label_pos = (x_pixel, y_pixel)

                if label_pos:
                    # GT Color matches object type
                    label_color = (255, 255, 0) if point_type == "object" else (
                        0, 255, 255)
                    # draw.text uses RGB
                    # Yellow/Cyan
                    pil_color = (label_color[0],
                                 label_color[1], label_color[2])

                    draw.text((label_pos[0] + 10, label_pos[1] - 10), description,
                              font=font, fill=pil_color, stroke_width=1, stroke_fill=(0, 0, 0))

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
