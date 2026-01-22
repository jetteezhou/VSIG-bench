import sys
import os
import time

# Ensure we are in the project directory
project_dir = "/home/tione/notebook/workspace/jetteezhou/code_space/VSIG-Bench"
os.chdir(project_dir)
sys.path.append(project_dir)

# Import Config to modify it
from config import Config

# Configure for Qwen3-Omni (Local vLLM)
Config.MODEL_PROVIDER = "openai"
Config.OPENAI_BASE_URL = "http://localhost:8005/v1"
Config.OPENAI_API_KEY = "EMPTY"  # vLLM allows any key or empty
Config.MODEL_NAME = "Qwen3-Omni-30B-A3B-Instruct" 

# Enable video input
Config.USE_VIDEO_INPUT = True

# Dataset path
Config.DATA_ROOT_DIR = os.path.join(project_dir, "data_new")

# Output directory
Config.OUTPUT_DIR = "results/Qwen3omni-30B-preview"
Config.SAVE_LOG = True

# Increase workers for efficiency if server handles it
# vLLM can handle concurrent requests but GPU mem is limit. 
# Safe start with small number of workers to avoid OOM or timeout.
Config.NUM_WORKERS = 4 

# Evaluation settings (Keep existing DeepSeek config for evaluation if keys are valid)
# Assuming keys in config.py are valid or env vars are set.
# If not, execution might fail at eval stage, but inference will be done.

print(f"Starting Evaluation with Config:")
print(f"Model: {Config.MODEL_NAME}")
print(f"Provider: {Config.MODEL_PROVIDER}")
print(f"URL: {Config.OPENAI_BASE_URL}")
print(f"Input: Video File mode = {Config.USE_VIDEO_INPUT}")
print(f"Data: {Config.DATA_ROOT_DIR}")
print("-" * 50)

import main

if __name__ == "__main__":
    main.main()
