# config.py
import os


class Config:
    # --- 模型配置 ---
    OPENAI_API_KEY = os.getenv(
        "OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv(
        "GEMINI_API_KEY", "")
    MODEL_PROVIDER = os.getenv(
        "MODEL_PROVIDER", "gemini")  # "openai" 或 "gemini"
    OPENAI_BASE_URL = os.getenv(
        "OPENAI_BASE_URL", "")
    MODEL_NAME = "gemini-3-pro-preview"

    # --- 多模型配置 ---
    # 如果配置了 MODELS 列表，将遍历所有模型进行处理
    # 每个模型配置包含以下字段：
    #   - provider: "openai" 或 "gemini"
    #   - name: 模型名称
    #   - api_key: API密钥（可选，如果未提供则使用全局配置）
    #   - base_url: OpenAI API的base_url（仅openai需要，可选）
    #   - coord_order: 坐标顺序 "xy" 或 "yx"（可选，默认根据provider自动设置）
    #   - use_video_input: 是否使用视频输入（可选，默认使用全局配置）
    MODELS = [
        # 示例配置：
        {
            "provider": "gemini",
            "name": "gemini-3-flash-preview",
            "api_key": None,  # None表示使用全局GEMINI_API_KEY
            "coord_order": "yx",
            "use_video_input": True
        },
        # {
        #     "provider": "gemini",
        #     "name": "gemini-3-pro-preview",
        #     "api_key": None,  # None表示使用全局GEMINI_API_KEY
        #     "coord_order": "yx",
        #     "use_video_input": True
        # },
        # {
        #     "provider": "gemini",
        #     "name": "gemini-2.5-flash",
        #     "api_key": None,  # None表示使用全局GEMINI_API_KEY
        #     "coord_order": "yx",
        #     "use_video_input": True
        # },
        # {
        #     "provider": "gemini",
        #     "name": "gemini-2.5-pro",
        #     "api_key": None,  # None表示使用全局GEMINI_API_KEY
        #     "coord_order": "yx",
        #     "use_video_input": True
        # },
        # {
        #     "provider": "openai",
        #     "name": "gpt-4o",
        #     "api_key": None,  # None表示使用全局OPENAI_API_KEY
        #     "base_url": None,  # None表示使用全局OPENAI_BASE_URL
        #     "coord_order": "xy",
        #     "use_video_input": False
        # },
    ]

    # --- 坐标顺序配置 ---
    # 坐标顺序："xy" 表示 [x, y]（默认），"yx" 表示 [y, x]
    # Gemini 模型默认使用 "yx"，其他模型默认使用 "xy"
    COORD_ORDER = os.getenv(
        "COORD_ORDER", "yx" if MODEL_PROVIDER == "gemini" else "xy")

    # --- 数据路径配置 ---
    DATA_ROOT_DIR = "data_new"  # 数据根目录，会遍历其下所有子目录（支持多个数据集）
    # 如果设置为 "data"，会自动扫描 data/ 下所有包含指令文件夹的子目录

    # --- 输出配置 ---
    OUTPUT_DIR = f"results/{MODEL_NAME}"  # 结果保存目录
    SAVE_LOG = False  # 是否保存日志文件
    NUM_FRAMES = 15  # 视频抽帧数量

    # 输入模式配置
    # True: 直接使用视频文件作为输入 (需要模型支持，如 Gemini、Qwenomni等)
    # False: 使用抽帧图像列表作为输入 (适用于所有模型)
    USE_VIDEO_INPUT = True

    # --- 并行推理配置 ---
    NUM_WORKERS = 20  # 并行推理的线程数，默认为4

    # --- 并行评估配置 ---
    EVAL_NUM_WORKERS = 10  # 并行评估的线程数，None 表示单线程

    # --- 评估模型配置（如果未配置，则使用推理模型进行评估）---
    # "openai" 或 "gemini"，None 表示使用推理模型
    EVAL_MODEL_PROVIDER = "gemini"
    EVAL_MODEL_NAME = "gemini-3-pro-preview"
    EVAL_GEMINI_API_KEY = os.getenv(
        "GEMINI_API_KEY", "")
    EVAL_OPENAI_API_KEY = os.getenv(
        "OPENAI_API_KEY", "")
    EVAL_OPENAI_BASE_URL = os.getenv(
        "OPENAI_BASE_URL", "")
