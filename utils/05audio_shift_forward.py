import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import shutil
import json

def shift_audio_forward(input_path, output_path, shift_seconds=1.0):
    """
    将音频轨道提前 shift_seconds 秒（即从原音频的第 1 秒开始提取，放在新视频的 0 秒处）。
    视频轨道保持不变。
    """
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-filter_complex', f'[0:a]atrim=start={shift_seconds},asetpts=PTS-STARTPTS[a]',
        '-map', '0:v', '-map', '[a]',
        '-c:v', 'copy', '-c:a', 'aac',
        '-y', str(output_path)
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        # 有些视频可能没有音轨，导致 atrim 失败，这里简单处理
        # 如果失败了，直接复制原文件
        shutil.copy2(input_path, output_path)

def process_json_timestamps(input_path, output_path, shift_ms=1000):
    """
    处理 JSON 文件中的时间戳，提前 shift_ms 毫秒。
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 判断是 annotations.json 还是 eval_gt.json
    filename = Path(input_path).name
    
    if filename == 'annotations.json':
        # data 是一个列表，每个元素是一个视频的标注
        if isinstance(data, list):
            for video_entry in data:
                if not isinstance(video_entry, dict): continue
                
                # 修改 object_space
                object_space = video_entry.get('object_space')
                if isinstance(object_space, list):
                    for obj in object_space:
                        if not isinstance(obj, dict): continue
                        if obj.get('asr_begin_time') is not None:
                            obj['asr_begin_time'] = max(0, obj['asr_begin_time'] - shift_ms)
                        if obj.get('asr_end_time') is not None:
                            obj['asr_end_time'] = max(0, obj['asr_end_time'] - shift_ms)
                
                # 修改 asr_result.words
                asr_res = video_entry.get('asr_result')
                if isinstance(asr_res, dict):
                    words = asr_res.get('words')
                    if isinstance(words, list):
                        for word in words:
                            if not isinstance(word, dict): continue
                            if word.get('begin_time') is not None:
                                word['begin_time'] = max(0, word['begin_time'] - shift_ms)
                            if word.get('end_time') is not None:
                                word['end_time'] = max(0, word['end_time'] - shift_ms)
                        
    elif filename == 'eval_gt.json':
        # data 是一个字典，key 是视频名
        if isinstance(data, dict):
            for video_name, content in data.items():
                if not isinstance(content, dict): continue
                answers = content.get('answer')
                if isinstance(answers, list):
                    for ans in answers:
                        if not isinstance(ans, dict): continue
                        if ans.get('asr_begin_time') is not None:
                            ans['asr_begin_time'] = max(0, ans['asr_begin_time'] - shift_ms)
                        if ans.get('asr_end_time') is not None:
                            ans['asr_end_time'] = max(0, ans['asr_end_time'] - shift_ms)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    base_dir = Path(__file__).parent.parent
    src_dir = base_dir / 'data_new'
    dst_dir = base_dir / 'data_audio_shift'
    
    if not src_dir.exists():
        print(f"Source directory {src_dir} does not exist.")
        return

    # 获取所有需要处理的文件
    all_files = list(src_dir.rglob('*'))
    
    for item in tqdm(all_files, desc="Processing files"):
        # 计算相对于 data_new 的路径，以便在 data_audio_shift 中重建结构
        relative_path = item.relative_to(src_dir)
        target_path = dst_dir / relative_path
        
        if item.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
            
        # 确保目标目录存在
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        if item.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # 处理视频文件：平移音频
            shift_audio_forward(item, target_path)
        elif item.name in ['annotations.json', 'eval_gt.json']:
            # 处理 JSON 文件中的时间戳
            process_json_timestamps(item, target_path)
        else:
            # 其他文件：直接复制
            shutil.copy2(item, target_path)

if __name__ == "__main__":
    main()
