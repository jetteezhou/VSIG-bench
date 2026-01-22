# config/settings.py
import os

# DashScope API配置 (用于ASR语音识别)
DASHSCOPE_API_KEY = os.getenv(
    "DASHSCOPE_API_KEY", "sk-dee7a03925be4cfb8fd61b0c1013dd34")

# ASR模型配置
ASR_MODEL = "fun-asr-realtime"

# 音频处理配置
AUDIO_SAMPLE_RATE = 16000  # 采样率 (Hz)
AUDIO_FORMAT = "wav"       # 音频格式（与官方示例保持一致，支持 wav/mp3/pcm）
AUDIO_CHANNELS = 1         # 单声道
