"""音频处理模块"""
from utils.settings import AUDIO_SAMPLE_RATE, AUDIO_FORMAT, AUDIO_CHANNELS, ASR_MODEL, DASHSCOPE_API_KEY
import subprocess
import tempfile
import json
from typing import List, Dict, Tuple, Optional
from http import HTTPStatus
from pydub import AudioSegment
import dashscope
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import sys

# 添加项目根目录到sys.path，以便导入utils模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_audio_and_video(video_path: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从视频中分离音频，返回音频路径和视频路径（这里视频路径直接返回原路径）
    """
    try:
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]

        # 音频输出路径（使用配置的格式）
        audio_output = os.path.join(
            output_dir, f"{base_name}.{AUDIO_FORMAT.lower()}")

        # 使用 ffmpeg 提取音频
        # 根据格式选择编码器
        if AUDIO_FORMAT.lower() == "wav":
            acodec = "pcm_s16le"  # WAV 格式使用 PCM
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", acodec,
                "-ar", str(AUDIO_SAMPLE_RATE),
                "-ac", str(AUDIO_CHANNELS),
                audio_output
            ]
        else:
            # MP3 格式
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                audio_output
            ]

        # 简单检查是否已存在，如果存在且非空则跳过? 不，为了安全还是覆盖吧，或者 ffmpeg -y 已经覆盖了。

        print(f"正在提取音频: {video_path} -> {audio_output}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)

        if os.path.exists(audio_output) and os.path.getsize(audio_output) > 0:
            # Return None for video_path to avoid accidental deletion of source
            return audio_output, None
        else:
            return None, None

    except Exception as e:
        print(f"提取音频失败: {e}")
        return None, None


def convert_to_mono(input_file: str, output_file: Optional[str] = None) -> Optional[str]:
    """
    将音频文件转换为单声道16kHz的MP3格式

    Args:
        input_file: 输入音频文件路径
        output_file: 输出文件路径，如果为None则创建临时文件

    Returns:
        转换后的文件路径，失败时返回None
    """
    try:
        print(f"正在加载音频文件: {input_file}")
        audio = AudioSegment.from_file(input_file)

        # 检查音频时长
        duration_ms = len(audio)
        if duration_ms == 0:
            print(f"错误：音频文件时长为0")
            return None

        print(f"音频时长: {duration_ms}ms ({duration_ms/1000:.2f}秒)")

        mono_audio = audio.set_channels(AUDIO_CHANNELS)
        mono_audio = mono_audio.set_frame_rate(AUDIO_SAMPLE_RATE)

        if output_file is None:
            # 根据配置的格式创建临时文件
            suffix = f'.{AUDIO_FORMAT.lower()}'
            temp_fd, output_file = tempfile.mkstemp(suffix=suffix)
            os.close(temp_fd)

        print(f"正在导出音频到: {output_file}")
        mono_audio.export(output_file, format=AUDIO_FORMAT)
        print(f"音频转换成功: {output_file}")
        return output_file

    except Exception as e:
        import traceback
        print(f"音频转换错误: {e}")
        print(f"详细错误信息: {traceback.format_exc()}")
        return None


def audio_to_words_with_timestamps(
    audio_file_path: str,
    api_key: Optional[str] = None
) -> Tuple[bool, List[Dict[str, any]], Optional[str]]:
    """
    将音频文件转换为带时间戳的词汇列表

    Args:
        audio_file_path: 音频文件路径
        api_key: DashScope API密钥，如果为None则使用 config/settings.py 中的 DASHSCOPE_API_KEY

    Returns:
        (识别是否成功, 词汇列表, 错误信息)
        词汇列表每个元素包含: text, begin_time, end_time
        错误信息：如果失败，返回详细的错误描述；成功则为None
    """
    mono_audio_file = None

    try:
        # 检查文件是否存在
        if not os.path.exists(audio_file_path):
            error_msg = f"音频文件不存在: {audio_file_path}"
            print(f"错误：{error_msg}")
            return False, [], error_msg

        # 检查文件大小
        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            error_msg = f"音频文件为空: {audio_file_path}"
            print(f"错误：{error_msg}")
            return False, [], error_msg

        print(f"音频文件大小: {file_size} 字节")

        # 设置API密钥
        dashscope.api_key = "sk-dee7a03925be4cfb8fd61b0c1013dd34"  # DASHSCOPE_API_KEY

        # 设置 WebSocket API URL（北京地域）
        # 若使用新加坡地域的模型，需将url替换为：wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference
        # wss://dashscope.aliyuncs.com/api-ws/v1/inference
        dashscope.base_websocket_api_url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

        # 转换音频为单声道
        print(f"开始转换音频格式: {audio_file_path}")
        mono_audio_file = convert_to_mono(audio_file_path)
        if mono_audio_file is None:
            error_msg = f"音频转换失败，请检查音频文件格式是否支持: {audio_file_path}"
            print(f"错误：{error_msg}")
            return False, [], error_msg

        # 检查转换后的文件
        if not os.path.exists(mono_audio_file):
            error_msg = "音频转换后文件不存在"
            print(f"错误：{error_msg}")
            return False, [], error_msg

        mono_file_size = os.path.getsize(mono_audio_file)
        if mono_file_size == 0:
            error_msg = "音频转换后文件为空"
            print(f"错误：{error_msg}")
            return False, [], error_msg

        print(f"已将音频转换为单声道: {mono_audio_file} (大小: {mono_file_size} 字节)")

        # 创建识别对象
        from dashscope.audio.asr import Recognition
        print(
            f"使用ASR模型: {ASR_MODEL}, 格式: {AUDIO_FORMAT}, 采样率: {AUDIO_SAMPLE_RATE}")
        recognition = Recognition(
            model=ASR_MODEL,
            format=AUDIO_FORMAT,
            sample_rate=AUDIO_SAMPLE_RATE,
            callback=None
        )

        # 执行语音识别
        print("开始调用语音识别API...")
        result = recognition.call(mono_audio_file)

        if result.status_code == HTTPStatus.OK:
            print('识别成功')
            sentence = result.get_sentence()

            # 详细的调试信息
            print(f"调试信息: sentence类型={type(sentence)}, sentence值={sentence}")
            if sentence is not None:
                print(
                    f"调试信息: sentence长度={len(sentence) if hasattr(sentence, '__len__') else 'N/A'}")
                if len(sentence) > 0:
                    print(
                        f"调试信息: sentence[0]类型={type(sentence[0])}, sentence[0]内容={sentence[0]}")
                    if isinstance(sentence[0], dict):
                        print(
                            f"调试信息: sentence[0]的键={list(sentence[0].keys())}")

            # 遍历所有sentence，收集所有有words的sentence的words
            words_list = []
            sentences_with_words = 0
            total_sentences = len(sentence) if sentence else 0

            if sentence and len(sentence) > 0:
                for idx, sent in enumerate(sentence):
                    if isinstance(sent, dict) and "words" in sent:
                        sent_words = sent.get("words", [])
                        if sent_words and len(sent_words) > 0:
                            # 验证words格式：每个word应该有text字段
                            valid_words = []
                            for word in sent_words:
                                if isinstance(word, dict) and "text" in word and word.get("text"):
                                    # 确保word有必要的字段
                                    word_dict = {
                                        "text": word.get("text", ""),
                                        "begin_time": word.get("begin_time", 0),
                                        "end_time": word.get("end_time", 0)
                                    }
                                    # 保留其他字段（如果有）
                                    if "punctuation" in word:
                                        word_dict["punctuation"] = word.get(
                                            "punctuation")
                                    valid_words.append(word_dict)

                            if valid_words:
                                words_list.extend(valid_words)
                                sentences_with_words += 1
                                print(
                                    f"调试信息: sentence[{idx}]包含{len(valid_words)}个有效词汇")
                            else:
                                print(
                                    f"调试信息: sentence[{idx}]的words格式不正确（缺少text字段或text为空）")
                        else:
                            print(f"调试信息: sentence[{idx}]的words为空或长度为0")
                    else:
                        print(f"调试信息: sentence[{idx}]格式不正确或缺少words键")

            # 如果至少有一个sentence有words，就认为识别成功
            if words_list and len(words_list) > 0:
                print(
                    f"调试信息: 总共收集到{len(words_list)}个词汇（来自{sentences_with_words}/{total_sentences}个sentence）")

                # 打印识别指标
                print(f'[Metric] requestId: {recognition.get_last_request_id()}, '
                      f'first package delay ms: {recognition.get_first_package_delay()}, '
                      f'last package delay ms: {recognition.get_last_package_delay()}')

                return True, words_list, None
            else:
                # 构建详细的错误信息
                error_details = []
                if sentence is None:
                    error_details.append("sentence为None")
                elif len(sentence) == 0:
                    error_details.append("sentence为空列表")
                elif len(sentence) > 0:
                    if sentences_with_words == 0:
                        error_details.append(
                            f"所有{len(sentence)}个sentence的words都为空或格式不正确")
                    else:
                        error_details.append(
                            f"虽然有{sentences_with_words}个sentence包含words，但提取后words_list为空")
                else:
                    error_details.append("sentence格式不正确")

                error_msg = f"识别结果为空：API返回成功但未识别到任何词汇 ({'; '.join(error_details)})"
                print(f"警告：{error_msg}")
                if sentence and len(sentence) > 0:
                    print(f"完整返回内容: 共{len(sentence)}个sentence")
                    import json
                    try:
                        # 只打印前3个sentence的详细信息，避免日志过长
                        preview = sentence[:3] if len(
                            sentence) > 3 else sentence
                        print(
                            f"前{len(preview)}个sentence的JSON格式: {json.dumps(preview, ensure_ascii=False, indent=2)}")
                    except:
                        print(f"sentence无法序列化为JSON")
                return False, [], error_msg
        else:
            error_msg = f"语音识别API调用失败 (状态码: {result.status_code}): {result.message}"
            print(f'错误：{error_msg}')
            # 尝试获取更详细的错误信息
            error_code = None
            request_id = None
            if hasattr(result, 'code'):
                error_code = result.code
                error_msg += f", 错误代码: {error_code}"
            if hasattr(result, 'request_id'):
                request_id = result.request_id
                error_msg += f", 请求ID: {request_id}"

            # 添加详细的诊断信息
            if result.status_code == 44:
                print("\n" + "="*60)
                print("⚠️  状态码 44 错误诊断")
                print("="*60)
                print(f"错误代码: {error_code}")
                print(f"请求ID: {request_id}")
                print("\n可能的原因：")
                print("1. 账户欠费 - 请检查 DashScope 控制台的账户余额")
                print("2. API Key 无效或过期 - 请检查 API Key 是否正确")
                print("3. API Key 权限不足 - 请确认该 Key 有 ASR 服务权限")
                print("4. ASR 服务未开通 - 请在 DashScope 控制台开通语音识别服务")
                print("5. 账户被限制 - 请联系 DashScope 客服检查账户状态")
                print("\n排查步骤：")
                print("1. 登录 https://dashscope.console.aliyun.com/")
                print("2. 检查账户余额和服务状态")
                print("3. 确认已开通 '语音识别' 服务")
                print("4. 检查 API Key 是否有效且有相应权限")
                print("5. 尝试创建一个新的 API Key")
                print("="*60)

            return False, [], error_msg

    except FileNotFoundError as e:
        error_msg = f"文件未找到: {str(e)}"
        print(f"错误：{error_msg}")
        return False, [], error_msg
    except PermissionError as e:
        error_msg = f"文件权限错误: {str(e)}"
        print(f"错误：{error_msg}")
        return False, [], error_msg
    except Exception as e:
        import traceback
        error_msg = f"处理音频时发生未知错误: {str(e)}\n详细错误: {traceback.format_exc()}"
        print(f"错误：{error_msg}")
        return False, [], error_msg

    finally:
        # 清理临时文件
        if mono_audio_file and os.path.exists(mono_audio_file):
            try:
                os.remove(mono_audio_file)
                print(f"已清理临时文件: {mono_audio_file}")
            except Exception as e:
                print(f"清理临时文件时出错: {e}")


def validate_api_key(api_key: str) -> Tuple[bool, Optional[str]]:
    """
    验证 API Key 的基本格式

    Returns:
        (是否有效, 错误信息)
    """
    if not api_key:
        return False, "API Key 未设置"

    if api_key.startswith("sk-xxx") or len(api_key) < 10:
        return False, "API Key 格式无效（看起来像是占位符）"

    if not api_key.startswith("sk-"):
        return False, "API Key 格式不正确（应以 'sk-' 开头）"

    return True, None


def _process_single_video_item(task_info: Dict) -> Dict:
    """
    处理单个视频项的辅助函数，用于线程池并行处理

    Args:
        task_info: 包含任务信息的字典，包括 video_path, video_name, root, api_key, template_asr, item_index

    Returns:
        处理结果字典，包含 success, item_index, asr_result, error 等字段
    """
    video_path = task_info['video_path']
    video_name = task_info['video_name']
    root = task_info['root']
    api_key = task_info['api_key']
    item_index = task_info['item_index']

    try:
        print(f"[线程] 处理视频: {video_name} (在 {os.path.basename(root)})")

        # Extract audio
        audio_path, _ = extract_audio_and_video(video_path, output_dir=root)
        if not audio_path:
            return {
                'success': False,
                'item_index': item_index,
                'error': '音频提取失败',
                'should_stop': False
            }

        # Recognize
        success, words_list, error = audio_to_words_with_timestamps(
            audio_path, api_key)

        # Clean up temp audio
        if os.path.exists(audio_path):
            os.remove(audio_path)

        if success:
            # Construct result MATCHING TEMPLATE
            full_text = "".join([w['text'] for w in words_list])
            new_asr_result = {
                "text": full_text,
                "words": words_list
            }
            print(f"[线程] ✓ 成功生成识别结果: {video_name} (文本长度: {len(full_text)})")
            return {
                'success': True,
                'item_index': item_index,
                'asr_result': new_asr_result,
                'error': None,
                'should_stop': False
            }
        else:
            error_str = str(error).lower()
            should_stop = False

            # 检查是否是严重的账户/认证问题
            if ("状态码: 44" in str(error) or
                    ("arrearage" in error_str and "状态码: 44" in str(error))):
                should_stop = True
                print("\n" + "="*60)
                print("⚠️  检测到状态码 44 错误，脚本停止运行。")
                print("="*60)
                print("这通常表示账户或服务配置问题，而非单个文件的问题。")
                print("请按照上面的诊断信息进行排查。")
                print("="*60)
            elif ("authentication" in error_str or "apikey" in error_str or
                  "invalidapikey" in error_str):
                should_stop = True
                print("\n" + "="*60)
                print("⚠️  检测到 API 认证错误，脚本停止运行。")
                print("错误类型: API Key 认证失败")
                print("解决方案: 请检查 API Key 是否正确设置。")
                print("="*60)

            print(f"[线程] ✗ 识别失败: {video_name}, 错误: {error}")
            return {
                'success': False,
                'item_index': item_index,
                'error': error,
                'should_stop': should_stop
            }

    except Exception as e:
        print(f"[线程] 处理异常: {video_name}, 错误: {e}")
        return {
            'success': False,
            'item_index': item_index,
            'error': f'处理异常: {str(e)}',
            'should_stop': False
        }


def fill_missing_asr_in_dir(target_dir: str, api_key: str) -> None:
    """
    递归扫描目录，查找没有 ASR 结果的视频并填充（使用50个线程并行处理）
    """
    print(f"开始扫描目录: {target_dir}")

    # 0. 验证 API Key
    print("正在验证 API Key...")
    is_valid, error_msg = validate_api_key(api_key)
    if not is_valid:
        print(f"错误: {error_msg}，请检查 config/settings.py")
        print("\n提示：")
        print("1. 确保 API Key 格式正确（应以 'sk-' 开头）")
        print("2. 从 DashScope 控制台获取有效的 API Key")
        print("3. 检查环境变量 DASHSCOPE_API_KEY 或配置文件中的设置")
        return

    print(f"✓ API Key 格式验证通过（Key 前缀: {api_key[:10]}...）")

    # 1. 学习现有格式
    print("正在扫描现有数据以学习 ASR 格式...")
    template_asr = None
    for root, dirs, files in os.walk(target_dir):
        if "annotations.json" in files:
            try:
                with open(os.path.join(root, "annotations.json"), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    asr = item.get("asr_result")
                    if asr and isinstance(asr, dict) and len(asr.get("words", [])) > 0:
                        print(f"找到格式模板参考: {item.get('video_name')}")
                        template_asr = asr
                        break
            except:
                pass
        if template_asr:
            break

    if not template_asr:
        print("警告: 未找到任何现有的有效 asr_result 样例，将使用默认标准格式。")
    else:
        print("已学习现有格式结构。")

    # 2. 收集所有需要处理的任务
    print("正在收集需要处理的任务...")
    tasks = []  # 存储所有任务信息
    json_files_data = {}  # 存储每个JSON文件的完整数据和路径

    for root, dirs, files in os.walk(target_dir):
        if "annotations.json" in files:
            json_path = os.path.join(root, "annotations.json")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                json_files_data[json_path] = {
                    'data': data,
                    'root': root
                }

                for item_index, item in enumerate(data):
                    # Check if processing is needed
                    needs_update = False
                    asr = item.get("asr_result")

                    # 检查是否需要更新：null、空字典、空列表，或者有text但为空且没有words
                    if asr is None:
                        needs_update = True
                    elif isinstance(asr, dict):
                        if not asr:  # 空字典
                            needs_update = True
                        elif "text" in asr and not asr.get("text") and not asr.get("words"):
                            needs_update = True
                    elif isinstance(asr, list) and not asr:  # 空列表
                        needs_update = True

                    if needs_update:
                        video_name = item.get("video_name")
                        if not video_name:
                            print(f"  警告: 第{item_index}项缺少video_name字段，跳过")
                            continue

                        # Find video file
                        video_path = os.path.join(root, video_name)
                        if not os.path.exists(video_path):
                            # Try common extensions if name doesn't have one or differs
                            found_vid = False
                            for ext in ['.mp4', '.mov', '.MOV', '.MP4']:
                                check_path = os.path.join(
                                    root, os.path.splitext(video_name)[0] + ext)
                                if os.path.exists(check_path):
                                    video_path = check_path
                                    found_vid = True
                                    break
                            if not found_vid:
                                print(f"  警告: 未找到视频文件 {video_name}，跳过")
                                continue  # Video not found, look for next item

                        tasks.append({
                            'video_path': video_path,
                            'video_name': video_name,
                            'root': root,
                            'api_key': api_key,
                            'template_asr': template_asr,
                            'item_index': item_index,
                            'json_path': json_path
                        })
                        print(f"  添加任务: {video_name} (索引: {item_index})")
            except Exception as e:
                print(f"无法读取/解析 JSON: {json_path}, 错误: {e}")

    if not tasks:
        print("没有找到需要处理的任务。")
        return

    print(f"共收集到 {len(tasks)} 个需要处理的任务，开始使用50个线程并行处理...")

    # 3. 使用线程池并行处理
    total_processed = 0
    total_errors = 0
    should_stop_global = False

    # 用于存储每个JSON文件的更新结果
    json_updates = {}  # json_path -> {item_index: asr_result}

    # 创建线程锁用于保护共享变量
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(_process_single_video_item, task): task
            for task in tasks
        }

        # 处理完成的任务
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()

                with lock:
                    if result['should_stop']:
                        should_stop_global = True

                    json_path = task['json_path']
                    if json_path not in json_updates:
                        json_updates[json_path] = {}

                    if result['success']:
                        json_updates[json_path][result['item_index']
                                                ] = result['asr_result']
                        total_processed += 1
                    else:
                        total_errors += 1

            except Exception as e:
                print(f"任务执行异常: {task['video_name']}, 错误: {e}")
                with lock:
                    total_errors += 1

            # 如果检测到需要停止的错误，取消剩余任务
            if should_stop_global:
                print("检测到严重错误，正在取消剩余任务...")
                for f in future_to_task:
                    f.cancel()
                break

    if should_stop_global:
        print("由于检测到严重错误，处理已停止。")
        return

    # 4. 更新JSON文件
    print("\n正在更新JSON文件...")
    total_files_updated = 0
    for json_path, updates in json_updates.items():
        if not updates:
            print(f"跳过文件 {json_path}：没有更新项")
            continue

        try:
            data = json_files_data[json_path]['data']
            modified = False
            update_count = 0

            print(f"处理文件: {json_path}，共有 {len(updates)} 个更新项")
            for item_index, asr_result in updates.items():
                if item_index < len(data):
                    old_asr = data[item_index].get("asr_result")
                    data[item_index]["asr_result"] = asr_result
                    modified = True
                    update_count += 1
                    video_name = data[item_index].get("video_name", "未知")
                    print(
                        f"  更新索引 {item_index} ({video_name}): null -> 已填充ASR结果")
                else:
                    print(f"  警告: 索引 {item_index} 超出范围（数据长度: {len(data)}）")

            if modified:
                print(f"保存更新到文件: {json_path}（更新了 {update_count} 项）")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                total_files_updated += 1
            else:
                print(f"警告: 文件 {json_path} 没有实际更新")
        except Exception as e:
            import traceback
            print(f"更新JSON文件失败: {json_path}, 错误: {e}")
            print(f"详细错误: {traceback.format_exc()}")

    print("\n" + "="*30)
    print(f"处理完成。")
    print(f"成功处理: {total_processed} 个视频")
    print(f"失败数量: {total_errors}")
    print(f"更新文件数: {total_files_updated} 个JSON文件")


if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="批量填充video_dir下所有annotations.json中缺失的ASR结果（使用50个线程并行处理）"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="data_new-2",
        help="要扫描的根目录路径（递归扫描所有annotations.json文件）"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="DashScope API密钥（如果不提供，将从config/settings.py中的DASHSCOPE_API_KEY读取）"
    )

    args = parser.parse_args()

    # 获取API密钥：优先使用命令行参数，否则使用配置文件中的key
    api_key = args.api_key if args.api_key else DASHSCOPE_API_KEY

    # 检查API密钥
    is_valid, error_msg = validate_api_key(api_key)
    if not is_valid:
        print(f"错误：{error_msg}")
        print("提示：请在config/settings.py中设置DASHSCOPE_API_KEY或使用--api_key参数")
        print("\n如何获取 API Key：")
        print("1. 访问 https://dashscope.console.aliyun.com/")
        print("2. 登录您的账户")
        print("3. 进入 'API-KEY管理' 页面")
        print("4. 创建或查看您的 API Key")
        exit(1)

    print(f"✓ API Key 格式验证通过（Key 前缀: {api_key[:10]}...）")

    # 检查目录是否存在
    if not os.path.exists(args.video_dir):
        print(f"错误：目录不存在: {args.video_dir}")
        exit(1)

    # 执行批量填充
    fill_missing_asr_in_dir(args.video_dir, api_key)
