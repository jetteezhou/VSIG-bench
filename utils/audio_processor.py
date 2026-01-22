"""音频处理模块"""
import subprocess
from utils.settings import AUDIO_SAMPLE_RATE, AUDIO_FORMAT, AUDIO_CHANNELS, ASR_MODEL, DASHSCOPE_API_KEY
import os
import tempfile
import json
from typing import List, Dict, Tuple, Optional
from http import HTTPStatus
from pydub import AudioSegment
import dashscope

import sys
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
        dashscope.api_key = DASHSCOPE_API_KEY

        # 设置 WebSocket API URL（北京地域）
        # 若使用新加坡地域的模型，需将url替换为：wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference
        dashscope.base_websocket_api_url = 'wss://dashscope.aliyuncs.com/api-ws/v1/inference'

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
            if sentence and len(sentence) > 0:
                for idx, sent in enumerate(sentence):
                    if isinstance(sent, dict) and "words" in sent:
                        sent_words = sent.get("words", [])
                        if sent_words and len(sent_words) > 0:
                            words_list.extend(sent_words)
                            print(
                                f"调试信息: sentence[{idx}]包含{len(sent_words)}个词汇")
                        else:
                            print(f"调试信息: sentence[{idx}]的words为空")
                    else:
                        print(f"调试信息: sentence[{idx}]格式不正确或缺少words键")

            # 如果至少有一个sentence有words，就认为识别成功
            if words_list and len(words_list) > 0:
                print(f"调试信息: 总共收集到{len(words_list)}个词汇")

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
                    error_details.append(
                        f"所有{len(sentence)}个sentence的words都为空或格式不正确")
                else:
                    error_details.append("sentence格式不正确")

                error_msg = f"识别结果为空：API返回成功但未识别到任何词汇 ({'; '.join(error_details)})"
                print(f"警告：{error_msg}")
                print(f"完整返回内容: sentence={sentence}")
                if sentence and len(sentence) > 0:
                    import json
                    try:
                        print(
                            f"所有sentence的JSON格式: {json.dumps(sentence, ensure_ascii=False, indent=2)}")
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


def print_words_with_timestamps(words_list: List[Dict[str, any]]) -> None:
    """
    打印带时间戳的词汇列表

    Args:
        words_list: 词汇列表
    """
    print("\n识别结果（词汇 | 开始时间 | 结束时间）：")
    print("-" * 50)
    for word in words_list:
        print(f"{word['text']} | {word['begin_time']}ms | {word['end_time']}ms")


def save_recognition_result_to_json(
    words_list: List[Dict[str, any]],
    output_file: str,
    source_file: Optional[str] = None,
    metadata: Optional[Dict[str, any]] = None
) -> bool:
    """
    将识别结果保存到JSON文件

    Args:
        words_list: 词汇列表
        output_file: 输出JSON文件路径
        source_file: 源文件路径（音频或视频文件）
        metadata: 额外的元数据信息

    Returns:
        是否保存成功
    """
    try:
        # 构建结果数据结构
        result = {
            "source_file": source_file,
            "total_words": len(words_list),
            "words": words_list
        }

        # 添加统计信息
        if words_list:
            first_word_time = words_list[0].get('begin_time', 0)
            last_word_time = words_list[-1].get('end_time', 0)
            duration_ms = last_word_time - first_word_time
            result["statistics"] = {
                "first_word_time_ms": first_word_time,
                "last_word_time_ms": last_word_time,
                "duration_ms": duration_ms,
                "duration_seconds": round(duration_ms / 1000.0, 2)
            }

        # 添加元数据
        if metadata:
            result["metadata"] = metadata

        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 保存到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 识别结果已保存到: {output_file}")
        return True

    except Exception as e:
        print(f"\n✗ 保存JSON文件失败: {str(e)}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False


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


def fill_missing_asr_in_dir(target_dir: str, api_key: str) -> None:
    """
    递归扫描目录，查找没有 ASR 结果的视频并填充
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

    total_processed = 0
    total_errors = 0

    # 2. 遍历处理
    for root, dirs, files in os.walk(target_dir):
        if "annotations.json" in files:
            json_path = os.path.join(root, "annotations.json")

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                modified = False

                for item in data:
                    # Check if processing is needed
                    needs_update = False
                    asr = item.get("asr_result")

                    if asr is None or (isinstance(asr, dict) and not asr) or (isinstance(asr, list) and not asr):
                        needs_update = True
                    elif isinstance(asr, dict) and "text" in asr and not asr["text"] and not asr.get("words"):
                        needs_update = True

                    if needs_update:
                        video_name = item.get("video_name")

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
                                continue  # Video not found, look for next item

                        print(
                            f"发现缺失 ASR 数据: {video_name} (在 {os.path.basename(root)})")

                        # Process
                        try:
                            # Extract audio
                            audio_path, _ = extract_audio_and_video(
                                video_path, output_dir=root)
                            if not audio_path:
                                print(f"  错误: 音频提取失败")
                                continue

                            # Recognize
                            success, words_list, error = audio_to_words_with_timestamps(
                                audio_path, api_key)

                            # Clean up temp audio
                            if os.path.exists(audio_path):
                                os.remove(audio_path)

                            if success:
                                # Construct result MATCHING TEMPLATE
                                full_text = "".join(
                                    [w['text'] for w in words_list])

                                new_asr_result = {
                                    "text": full_text,
                                    "words": words_list
                                }

                                # Try to mimic template
                                if template_asr:
                                    # Copy extra keys from template if any (simplified logic)
                                    pass

                                item["asr_result"] = new_asr_result
                                modified = True
                                total_processed += 1
                                print(f"  ✓ 成功生成识别结果 (文本长度: {len(full_text)})")
                            else:
                                print(f"  ✗ 识别失败: {error}")
                                total_errors += 1
                                # Strict error check for Auth and Account issues
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

                                if should_stop:
                                    return

                            # Rate limit protection (gentle)
                            import time
                            time.sleep(0.5)

                        except Exception as e:
                            print(f"  处理异常: {e}")
                            total_errors += 1

                if modified:
                    print(f"保存更新到文件: {json_path}")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"无法读取/解析 JSON: {json_path}, 错误: {e}")

    print("\n" + "="*30)
    print(f"处理完成。")
    print(f"成功更新: {total_processed} 个视频")
    print(f"失败数量: {total_errors}")


if __name__ == "__main__":
    import argparse
    import glob

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试音频识别处理（支持视频文件分离和音频识别）")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="包含mov/mp4视频文件的文件夹路径（如果指定，将处理文件夹内所有视频文件）"
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        default=None,
        help="要识别的音频文件路径（如果指定，将直接识别该音频文件）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（用于保存分离出的音频和视频文件，默认使用临时目录）"
    )
    parser.add_argument(
        "--keep_files",
        action="store_true",
        help="保留分离出的音频和视频文件（默认会删除临时文件）"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="DashScope API密钥（如果不提供，将从 config/settings.py 中的 DASHSCOPE_API_KEY 读取）"
    )
    parser.add_argument(
        "--json_output_dir",
        type=str,
        default=None,
        help="JSON结果输出目录（如果不指定，将保存到输出目录或源文件所在目录）"
    )
    parser.add_argument(
        "--fill_missing_asr",
        action="store_true",
        help="启用批量填充模式：递归扫描 --video_dir 下所有 annotations.json，填充缺失的 asr_result"
    )
    parser.add_argument(
        "--test_api_key",
        action="store_true",
        help="测试 API Key 是否有效（需要提供一个测试音频文件）"
    )

    args = parser.parse_args()

    # 获取API密钥：优先使用命令行参数，否则使用配置文件中的key
    api_key = DASHSCOPE_API_KEY

    # 检查API密钥
    is_valid, error_msg = validate_api_key(api_key)
    if not is_valid:
        print(f"错误：{error_msg}")
        print("提示：请在 config/settings.py 中设置 DASHSCOPE_API_KEY 或使用 --api_key 参数")
        print("\n如何获取 API Key：")
        print("1. 访问 https://dashscope.console.aliyun.com/")
        print("2. 登录您的账户")
        print("3. 进入 'API-KEY管理' 页面")
        print("4. 创建或查看您的 API Key")
        exit(1)

    print(f"✓ API Key 格式验证通过（Key 前缀: {api_key[:10]}...）")

    # 确定处理模式
    video_files = []
    audio_file_path = None

    if args.fill_missing_asr:
        if not args.video_dir:
            print("错误: 使用 --fill_missing_asr 必须指定 --video_dir (作为扫描的根目录)")
            exit(1)
        fill_missing_asr_in_dir(args.video_dir, api_key)
        exit(0)

    if args.video_dir:
        # 模式1：处理视频文件夹
        if not os.path.exists(args.video_dir):
            print(f"错误：视频文件夹不存在: {args.video_dir}")
            exit(1)

        # 查找所有 mov 和 mp4 文件
        video_files.extend(glob.glob(os.path.join(args.video_dir, "*.mov")))
        video_files.extend(glob.glob(os.path.join(args.video_dir, "*.mp4")))
        video_files.extend(glob.glob(os.path.join(args.video_dir, "*.MOV")))
        video_files.extend(glob.glob(os.path.join(args.video_dir, "*.MP4")))

        if not video_files:
            print(f"错误：在文件夹 {args.video_dir} 中未找到任何 mov 或 mp4 文件")
            exit(1)

        print(f"找到 {len(video_files)} 个视频文件")

    elif args.audio_file:
        # 模式2：直接处理音频文件
        audio_file_path = args.audio_file
        if not os.path.exists(audio_file_path):
            print(f"错误：音频文件不存在: {audio_file_path}")
            exit(1)
    else:
        print("错误：请指定 --video_dir 或 --audio_file 参数")
        parser.print_help()
        exit(1)

    # 处理视频文件
    if video_files:
        print("=" * 80)
        print("开始批量处理视频文件")
        print("=" * 80)

        # 确定输出目录
        output_dir = args.output_dir
        is_temp_dir = False
        if output_dir is None:
            if args.keep_files:
                # 如果没有指定输出目录但需要保留文件，使用视频文件夹
                output_dir = args.video_dir
            else:
                # 使用临时目录
                output_dir = tempfile.mkdtemp()
                is_temp_dir = True
                print(f"使用临时目录: {output_dir}")
        else:
            os.makedirs(output_dir, exist_ok=True)
            print(f"使用输出目录: {output_dir}")

        print(f"共找到 {len(video_files)} 个视频文件")
        print("-" * 80)

        # 确定JSON输出目录
        json_output_dir = args.json_output_dir
        if json_output_dir is None:
            # 如果没有指定，使用输出目录或视频文件夹
            json_output_dir = output_dir if not is_temp_dir else args.video_dir
        os.makedirs(json_output_dir, exist_ok=True)

        # 用于存储所有文件的识别结果
        all_results = []

        success_count = 0
        fail_count = 0

        for idx, video_file in enumerate(video_files, 1):
            print(
                f"\n[{idx}/{len(video_files)}] 处理视频文件: {os.path.basename(video_file)}")
            print("-" * 80)

            try:
                # 1. 分离音频和视频
                print(f"步骤1: 分离音频和视频...")
                audio_path, video_path = extract_audio_and_video(
                    video_file,
                    output_dir=output_dir
                )

                if audio_path is None or video_path is None:
                    print(f"✗ 分离失败: 无法从视频文件中提取音频或视频")
                    fail_count += 1
                    # 记录失败的文件
                    all_results.append({
                        "source_file": video_file,
                        "status": "failed",
                        "error": "无法从视频文件中提取音频或视频",
                        "total_words": 0,
                        "words": []
                    })
                    continue

                print(f"✓ 分离成功:")
                print(f"  音频文件: {audio_path}")
                print(f"  视频文件: {video_path}")

                # 2. 识别音频
                print(f"\n步骤2: 识别音频...")
                success, words_list, error_msg = audio_to_words_with_timestamps(
                    audio_path, api_key)

                if success:
                    print(f"\n✓ 识别成功！")
                    print_words_with_timestamps(words_list)

                    # 打印统计信息
                    total_words = len(words_list)
                    if total_words > 0:
                        first_word_time = words_list[0].get('begin_time', 0)
                        last_word_time = words_list[-1].get('end_time', 0)
                        duration_ms = last_word_time - first_word_time
                        print(f"\n统计信息:")
                        print(f"  识别词汇数: {total_words}")
                        print(
                            f"  音频时长: {duration_ms}ms ({duration_ms/1000:.2f}秒)")

                    # 收集识别结果
                    result_data = {
                        "source_file": video_file,
                        "status": "success",
                        "total_words": total_words,
                        "words": words_list,
                        "metadata": {
                            "video_file": video_file,
                            "audio_file": audio_path,
                            "video_file_extracted": video_path
                        }
                    }

                    # 添加统计信息
                    if total_words > 0:
                        first_word_time = words_list[0].get('begin_time', 0)
                        last_word_time = words_list[-1].get('end_time', 0)
                        duration_ms = last_word_time - first_word_time
                        result_data["statistics"] = {
                            "first_word_time_ms": first_word_time,
                            "last_word_time_ms": last_word_time,
                            "duration_ms": duration_ms,
                            "duration_seconds": round(duration_ms / 1000.0, 2)
                        }

                    all_results.append(result_data)
                    success_count += 1
                else:
                    print(f"\n✗ 识别失败: {error_msg}")
                    fail_count += 1
                    # 记录失败的文件
                    all_results.append({
                        "source_file": video_file,
                        "status": "failed",
                        "error": error_msg,
                        "total_words": 0,
                        "words": []
                    })

                # 4. 清理临时文件（如果需要）
                if not args.keep_files:
                    try:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                            print(f"已删除临时音频文件: {audio_path}")
                        if os.path.exists(video_path):
                            os.remove(video_path)
                            print(f"已删除临时视频文件: {video_path}")
                    except Exception as e:
                        print(f"清理临时文件时出错: {e}")

            except Exception as e:
                import traceback
                print(f"\n✗ 处理视频文件时发生错误: {str(e)}")
                print(f"详细错误: {traceback.format_exc()}")
                fail_count += 1
                # 记录失败的文件
                all_results.append({
                    "source_file": video_file,
                    "status": "failed",
                    "error": f"处理时发生异常: {str(e)}",
                    "total_words": 0,
                    "words": []
                })

        # 保存所有识别结果到同一个JSON文件
        if all_results:
            # 生成JSON文件名（基于视频文件夹名称）
            video_dir_basename = os.path.basename(
                os.path.abspath(args.video_dir))
            json_output_file = os.path.join(
                json_output_dir, f"{video_dir_basename}_recognition.json")

            # 构建汇总结果
            summary_result = {
                "video_dir": args.video_dir,
                "total_files": len(video_files),
                "success_count": success_count,
                "fail_count": fail_count,
                "results": all_results
            }

            # 保存到JSON文件
            try:
                with open(json_output_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_result, f, ensure_ascii=False, indent=2)
                print(f"\n✓ 所有识别结果已保存到: {json_output_file}")
            except Exception as e:
                print(f"\n✗ 保存汇总JSON文件失败: {str(e)}")
                import traceback
                print(f"详细错误: {traceback.format_exc()}")

        # 打印总结
        print("\n" + "=" * 80)
        print("处理完成")
        print("=" * 80)
        print(f"成功: {success_count} 个文件")
        print(f"失败: {fail_count} 个文件")
        print(f"总计: {len(video_files)} 个文件")
        if all_results:
            print(f"识别结果已保存到: {json_output_file}")

        if not args.keep_files and is_temp_dir:
            try:
                os.rmdir(output_dir)
                print(f"已清理临时目录: {output_dir}")
            except:
                pass

    # 处理单个音频文件
    elif audio_file_path:
        print("=" * 60)
        print("开始音频识别测试")
        print("=" * 60)
        print(f"音频文件: {audio_file_path}")
        print(f"API密钥: {'已设置' if api_key else '未设置'}")
        print("-" * 60)

        success, words_list, error_msg = audio_to_words_with_timestamps(
            audio_file_path, api_key)

        if success:
            print("\n✓ 识别成功！")
            print_words_with_timestamps(words_list)

            # 打印统计信息
            total_words = len(words_list)
            if total_words > 0:
                first_word_time = words_list[0].get('begin_time', 0)
                last_word_time = words_list[-1].get('end_time', 0)
                duration_ms = last_word_time - first_word_time
                print(f"\n统计信息:")
                print(f"  识别词汇数: {total_words}")
                print(f"  音频时长: {duration_ms}ms ({duration_ms/1000:.2f}秒)")

            # 保存识别结果到JSON文件
            json_output_dir = args.json_output_dir
            if json_output_dir is None:
                # 如果没有指定，使用音频文件所在目录
                json_output_dir = os.path.dirname(audio_file_path) or "."
            os.makedirs(json_output_dir, exist_ok=True)

            audio_basename = os.path.splitext(
                os.path.basename(audio_file_path))[0]
            json_output_file = os.path.join(
                json_output_dir, f"{audio_basename}_recognition.json")
            save_recognition_result_to_json(
                words_list,
                json_output_file,
                source_file=audio_file_path
            )
        else:
            print(f"\n✗ 识别失败: {error_msg}")
            exit(1)
