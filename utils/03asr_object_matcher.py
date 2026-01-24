import os
import sys
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.models.base_vlm import OpenAIVLM, GeminiVLM
from src.prompts.vsig_prompts import VSIGPrompts

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ASR_Matcher")

def get_vlm_model():
    """根据配置获取 VLM 模型实例"""
    if not Config.MODELS:
        raise ValueError("Config.MODELS 列表为空，请在 config.py 中配置模型。")
    
    m_cfg = Config.MODELS[0]
    provider = m_cfg.get("provider", "gemini")
    model_name = m_cfg.get("name", "gemini-3-flash-preview")
    api_key = m_cfg.get("api_key") or (Config.GEMINI_API_KEY if provider == "gemini" else Config.OPENAI_API_KEY)
    coord_order = m_cfg.get("coord_order", Config.COORD_ORDER)

    if provider == "gemini":
        return GeminiVLM(api_key=api_key, model_name=model_name, coord_order=coord_order)
    else:
        base_url = m_cfg.get("base_url") or Config.OPENAI_BASE_URL
        return OpenAIVLM(api_key=api_key, base_url=base_url, model_name=model_name, coord_order=coord_order)

def process_single_item(task):
    """处理单个任务项（线程执行函数）"""
    file_path = task['file_path']
    item_idx = task['item_idx']
    item = task['item']
    model = task['model']
    overwrite = task['overwrite']
    
    asr_result = item.get("asr_result")
    object_space = item.get("object_space", [])
    
    if not asr_result or not object_space:
        return file_path, item_idx, item, False

    # 检查是否已经处理过
    if not overwrite:
        already_processed = True
        for obj in object_space:
            if "asr_begin_time" not in obj:
                already_processed = False
                break
        if already_processed:
            return file_path, item_idx, item, False

    prompt = VSIGPrompts.get_asr_matching_prompt(asr_result, object_space)
    
    try:
        # 调用模型生成匹配结果
        response = model.generate(image_paths=[], prompt=prompt)
        
        match_results = []
        if isinstance(response, list):
            match_results = response
        elif isinstance(response, dict):
            for key in ["results", "data", "matches", "object_space"]:
                if key in response and isinstance(response[key], list):
                    match_results = response[key]
                    break
            if not match_results:
                if all(str(k).isdigit() for k in response.keys()):
                    match_results = [dict(v, index=int(k)) for k, v in response.items()]
        
        if not match_results:
            return file_path, item_idx, item, False

        modified = False
        for res in match_results:
            idx = res.get("index")
            if idx is not None and idx < len(object_space):
                obj = object_space[idx]
                obj["asr_begin_time"] = res.get("asr_begin_time")
                obj["asr_end_time"] = res.get("asr_end_time")
                obj["asr_match_error"] = res.get("asr_match_error", False)
                modified = True
        
        return file_path, item_idx, item, modified

    except Exception as e:
        logger.error(f"处理文件 {file_path} 中的项 {item_idx} 时出错: {e}")
        return file_path, item_idx, item, False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="使用 LLM 多线程匹配 ASR 结果与 object_space")
    parser.add_argument("--data_dir", type=str, default=Config.DATA_ROOT_DIR, help="数据目录")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖现有匹配结果")
    parser.add_argument("--workers", type=int, default=Config.NUM_WORKERS, help="并行线程数")
    args = parser.parse_args()

    model = get_vlm_model()
    
    # 1. 收集所有任务
    all_tasks = []
    file_data_map = {} # 用于存储原始文件数据，以便后续回写
    
    json_files = []
    for root, dirs, files in os.walk(args.data_dir):
        if "annotations.json" in files:
            json_files.append(os.path.join(root, "annotations.json"))
    
    logger.info(f"正在扫描文件...")
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                file_data_map[file_path] = data
                for i, item in enumerate(data):
                    all_tasks.append({
                        'file_path': file_path,
                        'item_idx': i,
                        'item': item,
                        'model': model,
                        'overwrite': args.overwrite
                    })
        except Exception as e:
            logger.error(f"加载文件 {file_path} 失败: {e}")

    logger.info(f"共收集到 {len(all_tasks)} 个待处理项，准备使用 {args.workers} 个线程进行并行处理...")

    # 2. 多线程执行任务
    modified_files = set()
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(process_single_item, task): task for task in all_tasks}
        
        for future in as_completed(future_to_task):
            file_path, item_idx, updated_item, is_modified = future.result()
            completed_count += 1
            
            if is_modified:
                file_data_map[file_path][item_idx] = updated_item
                modified_files.add(file_path)
            
            if completed_count % 10 == 0:
                logger.info(f"进度: {completed_count}/{len(all_tasks)}")

    # 3. 回写修改过的文件
    logger.info(f"处理完成，正在回写 {len(modified_files)} 个修改过的文件...")
    for file_path in modified_files:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(file_data_map[file_path], f, ensure_ascii=False, indent=2)
            logger.info(f"已更新: {file_path}")
        except Exception as e:
            logger.error(f"回写文件 {file_path} 失败: {e}")

    logger.info("所有任务处理完毕。")

if __name__ == "__main__":
    main()
