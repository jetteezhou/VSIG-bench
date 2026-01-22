import sys
import os
import time

# Ensure we are in the project directory
project_dir = "/home/tione/notebook/workspace/jetteezhou/code_space/VSIG-Bench"
os.chdir(project_dir)
sys.path.append(project_dir)

# Import Config to modify it
from config import Config

# Configure for HumanOmni (Local vLLM Service)
Config.MODEL_PROVIDER = "openai"
Config.OPENAI_BASE_URL = "http://localhost:8001/v1"
Config.OPENAI_API_KEY = "EMPTY"  # Local service does not check key
Config.MODEL_NAME = "HumanOmni-7B-Omni" 

# Enable video input (Direct File Mode)
Config.USE_VIDEO_INPUT = True

# Dataset path
Config.DATA_ROOT_DIR = os.path.join(project_dir, "data_new")

# Output directory
Config.OUTPUT_DIR = "results/HumanOmni-7B-Omni"
Config.SAVE_LOG = True

# Reduce workers to avoid OOM since we are running both model and eval locally
Config.NUM_WORKERS = 1

print(f"Starting HumanOmni Evaluation with Config:")
print(f"Model: {Config.MODEL_NAME}")
print(f"Provider: {Config.MODEL_PROVIDER}")
print(f"URL: {Config.OPENAI_BASE_URL}")
print(f"Input: Video File mode = {Config.USE_VIDEO_INPUT}")
print(f"Data: {Config.DATA_ROOT_DIR}")
print(f"Workers: {Config.NUM_WORKERS}")
print("-" * 50)

import main

if __name__ == "__main__":
    main.main()
