import os
import subprocess
import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_video(input_path, output_path, crf=41, fps=30):
    """
    使用优化的两遍 VP9 压缩视频。
    """
    if os.path.exists(output_path):
        logger.info(f"输出文件已存在，跳过: {output_path}")
        return

    logger.info(f"开始转换: {input_path} -> {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取输入文件的基础名称用于生成的 ffmpeg2pass 日志文件前缀
    # 使用进程 ID 避免并行运行时的冲突
    pass_log_file = os.path.join(os.path.dirname(output_path), f"passlog_{os.getpid()}")

    try:
        # 第一遍：-speed 8 (最高速)，只生成统计数据
        pass1_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libvpx-vp9",
            "-pass", "1",
            "-passlogfile", pass_log_file,
            "-b:v", "0",
            "-crf", str(crf),
            "-r", str(fps),
            "-speed", "8",
            "-row-mt", "1",
            "-tile-columns", "6",
            "-frame-parallel", "1",
            "-an",
            "-f", "null",
            "/dev/null"
        ]
        
        logger.info(f"[{os.getpid()}] 执行第一遍 (高速): {os.path.basename(input_path)}")
        subprocess.run(pass1_cmd, check=True, capture_output=True)

        # 第二遍：-speed 4 (均衡速度与质量)，生成最终文件
        pass2_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libvpx-vp9",
            "-pass", "2",
            "-passlogfile", pass_log_file,
            "-b:v", "0",
            "-crf", str(crf),
            "-r", str(fps),
            "-speed", "4",
            "-row-mt", "1",
            "-tile-columns", "6",
            "-frame-parallel", "1",
            "-auto-alt-ref", "1",
            "-lag-in-frames", "25",
            "-c:a", "aac",
            "-b:a", "128k",
            output_path
        ]
        
        logger.info(f"[{os.getpid()}] 执行第二遍 (均衡): {os.path.basename(input_path)}")
        subprocess.run(pass2_cmd, check=True, capture_output=True)
        
        logger.info(f"转换完成: {output_path}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg 出错 (文件: {input_path}, 退出码 {e.returncode})")
        raise
    finally:
        # 清理该进程生成的日志文件
        for suffix in ["-0.log", "-0.log.mbtree"]:
            f = f"{pass_log_file}{suffix}"
            if os.path.exists(f):
                os.remove(f)

def process_directory(input_dir, output_dir, crf=41, fps=30, workers=1):
    """
    批量处理目录中的所有视频，支持并行。
    """
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg')
    tasks = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".mp4")
                tasks.append((input_path, output_path, crf, fps))

    if not tasks:
        logger.info("未发现可处理的视频文件。")
        return

    logger.info(f"发现 {len(tasks)} 个任务，使用 {workers} 个工作进程进行处理...")
    
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务
            futures = [executor.submit(convert_video, *task) for task in tasks]
            # 等待所有任务完成并获取结果（用于捕获异常）
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"任务执行失败: {e}")
    else:
        for t in tasks:
            convert_video(*t)

def main():
    parser = argparse.ArgumentParser(description="视频格式转换和两遍 VP9 压缩工具 (CPU 加速版)")
    parser.add_argument("-i", "--input", default="data", help="输入文件或目录 (默认: data)")
    parser.add_argument("-o", "--output", default="data_new", help="输出文件或目录 (默认: data_new)")
    parser.add_argument("--crf", type=int, default=41, help="VP9 CRF 值 (默认: 41)")
    parser.add_argument("--fps", type=int, default=30, help="输出帧率 (默认: 30)")
    parser.add_argument("-w", "--workers", type=int, default=4, help="并行工作进程数 (默认: 1)")
    
    args = parser.parse_args()
    
    # ... (保持原有的 main 逻辑，调用带 workers 的 process_directory)
    
    if os.path.isdir(args.input):
        logger.info(f"处理目录: {args.input}")
        process_directory(args.input, args.output, args.crf, args.fps, args.workers)
    elif os.path.isfile(args.input):
        logger.info(f"处理单个文件: {args.input}")
        # ... (保持原有的单文件逻辑)
        if os.path.isdir(args.output):
            output_path = os.path.join(args.output, os.path.splitext(os.path.basename(args.input))[0] + ".mp4")
        else:
            output_path = args.output
            if not output_path.lower().endswith(".mp4"):
                output_path += ".mp4"
        convert_video(args.input, output_path, args.crf, args.fps)
    else:
        logger.error(f"输入路径不存在: {args.input}")
        sys.exit(1)

if __name__ == "__main__":
    main()

