"""音频处理模块"""
import os
import tempfile
import json
from typing import List, Dict, Tuple, Optional
from http import HTTPStatus
from pydub import AudioSegment
import dashscope

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import AUDIO_SAMPLE_RATE, AUDIO_FORMAT, AUDIO_CHANNELS, ASR_MODEL, DASHSCOPE_API_KEY
from pipeline.video_preprocessor import extract_audio_and_video


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
            temp_fd, output_file = tempfile.mkstemp(suffix='.mp3')
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
        
        # 检查API密钥：如果未提供，则使用配置文件中的默认值
        if not api_key:
            api_key = DASHSCOPE_API_KEY
        
        if not api_key:
            error_msg = "DashScope API密钥未设置，请在 config/settings.py 中设置 DASHSCOPE_API_KEY"
            print(f"错误：{error_msg}")
            return False, [], error_msg
        
        # 设置API密钥
        dashscope.api_key = api_key
        
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
        print(f"使用ASR模型: {ASR_MODEL}, 格式: {AUDIO_FORMAT}, 采样率: {AUDIO_SAMPLE_RATE}")
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
                print(f"调试信息: sentence长度={len(sentence) if hasattr(sentence, '__len__') else 'N/A'}")
                if len(sentence) > 0:
                    print(f"调试信息: sentence[0]类型={type(sentence[0])}, sentence[0]内容={sentence[0]}")
                    if isinstance(sentence[0], dict):
                        print(f"调试信息: sentence[0]的键={list(sentence[0].keys())}")
            
            # 遍历所有sentence，收集所有有words的sentence的words
            words_list = []
            if sentence and len(sentence) > 0:
                for idx, sent in enumerate(sentence):
                    if isinstance(sent, dict) and "words" in sent:
                        sent_words = sent.get("words", [])
                        if sent_words and len(sent_words) > 0:
                            words_list.extend(sent_words)
                            print(f"调试信息: sentence[{idx}]包含{len(sent_words)}个词汇")
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
                    error_details.append(f"所有{len(sentence)}个sentence的words都为空或格式不正确")
                else:
                    error_details.append("sentence格式不正确")
                
                error_msg = f"识别结果为空：API返回成功但未识别到任何词汇 ({'; '.join(error_details)})"
                print(f"警告：{error_msg}")
                print(f"完整返回内容: sentence={sentence}")
                if sentence and len(sentence) > 0:
                    import json
                    try:
                        print(f"所有sentence的JSON格式: {json.dumps(sentence, ensure_ascii=False, indent=2)}")
                    except:
                        print(f"sentence无法序列化为JSON")
                return False, [], error_msg
        else:
            error_msg = f"语音识别API调用失败 (状态码: {result.status_code}): {result.message}"
            print(f'错误：{error_msg}')
            # 尝试获取更详细的错误信息
            if hasattr(result, 'code'):
                error_msg += f", 错误代码: {result.code}"
            if hasattr(result, 'request_id'):
                error_msg += f", 请求ID: {result.request_id}"
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
    
    args = parser.parse_args()
    
    # 获取API密钥：优先使用命令行参数，否则使用配置文件中的key
    api_key = args.api_key or DASHSCOPE_API_KEY
    
    # 检查API密钥
    if not api_key:
        print("错误：DashScope API密钥未设置")
        print("提示：请在 config/settings.py 中设置 DASHSCOPE_API_KEY 或使用 --api_key 参数")
        exit(1)
    
    # 确定处理模式
    video_files = []
    audio_file_path = None
    
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
            print(f"\n[{idx}/{len(video_files)}] 处理视频文件: {os.path.basename(video_file)}")
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
                success, words_list, error_msg = audio_to_words_with_timestamps(audio_path, api_key)
                
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
                        print(f"  音频时长: {duration_ms}ms ({duration_ms/1000:.2f}秒)")
                    
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
            video_dir_basename = os.path.basename(os.path.abspath(args.video_dir))
            json_output_file = os.path.join(json_output_dir, f"{video_dir_basename}_recognition.json")
            
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
        
        success, words_list, error_msg = audio_to_words_with_timestamps(audio_file_path, api_key)
        
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
            
            audio_basename = os.path.splitext(os.path.basename(audio_file_path))[0]
            json_output_file = os.path.join(json_output_dir, f"{audio_basename}_recognition.json")
            save_recognition_result_to_json(
                words_list,
                json_output_file,
                source_file=audio_file_path
            )
        else:
            print(f"\n✗ 识别失败: {error_msg}")
            exit(1)