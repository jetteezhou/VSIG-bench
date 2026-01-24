import sys
import os
import time
import random
import re
import ast
import json
import logging
from openai import OpenAI

# Ensure we are in the project directory
project_dir = "/home/tione/notebook/workspace/jetteezhou/code_space/VSIG-Bench"
os.chdir(project_dir)
sys.path.append(project_dir)

# Import Config and BaseVLM
from config import Config
import src.models.base_vlm as base_vlm

# --- 1. Monkey Patch OpenAIVLM for Load Balancing ---
OriginalOpenAIVLM = base_vlm.OpenAIVLM

class MultiPortOpenAIVLM(OriginalOpenAIVLM):
    def __init__(self, api_key, base_url=None, **kwargs):
        # Handle list of URLs
        self._set_base_urls = base_url if isinstance(base_url, list) else [base_url]
        print(f"Initializing MultiPortOpenAIVLM with {len(self._set_base_urls)} endpoints")
        
        # Create a client for each URL
        self._clients = []
        for url in self._set_base_urls:
            try:
                self._clients.append(OpenAI(api_key=api_key, base_url=url))
            except Exception as e:
                print(f"Warning: Failed to create client for {url}: {e}")
        
        if not self._clients:
            raise RuntimeError("No available OpenAI clients could be created!")

        # Initialize parent
        super().__init__(api_key, base_url=self._set_base_urls[0], **kwargs)

    @property
    def client(self):
        return random.choice(self._clients)
    
    @client.setter
    def client(self, value):
        pass

    # --- 2. Monkey Patch _parse_json_response for Robustness (Here ONLY) ---
    def _parse_json_response(self, content):
        """
        Robust JSON parser specifically for HumanOmni's degenerated output.
        """
        logger = logging.getLogger("VSIG_Logger")
        content = content.strip()
        
        # Markdown cleanup
        if "```" in content:
            match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()
            else:
                if content.startswith("```"):
                    lines = content.split('\n')
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].strip() == "```":
                        lines = lines[:-1]
                    content = "\n".join(lines)

        # Standard try
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Regex Fallback Extraction
        try:
            fallback_result = {}
            
            # Extract explicit_command
            cmd_match = re.search(r'["\']?explicit_command["\']?\s*[:=]\s*["\'](.*?)["\']', content)
            if cmd_match:
                fallback_result["explicit_command"] = cmd_match.group(1)
            
            # Extract selected_options
            opts_match = re.search(r'["\']?selected_options["\']?\s*[:=]\s*\[(.*?)\]', content)
            if opts_match:
                raw_opts = opts_match.group(1)
                fallback_result["selected_options"] = [
                    opt.strip().strip("'").strip('"') 
                    for opt in raw_opts.replace("，", ",").split(",") 
                    if opt.strip()
                ]
            else:
                fallback_result["selected_options"] = []

            # Extract point_list (coordinates only)
            fallback_result["point_list"] = []
            # Match: point:[123, 456] or point [123 456]
            points = re.findall(r'point\s*[:=]?\s*\[\s*(-?\d+(?:\.\d+)?)(?:[,\s]+)(-?\d+(?:\.\d+)?)\s*\]', content, re.IGNORECASE)
            for x, y in points:
                fallback_result["point_list"].append({
                    "point": [float(x), float(y)],
                    "type": "target_object", 
                    "description": "extracted_by_regex"
                })
            
            if fallback_result.get("explicit_command") or fallback_result.get("point_list"):
                logger.warning(f"Recovered JSON via regex: {str(fallback_result)[:100]}...")
                return fallback_result
        except Exception as e:
            logger.error(f"Regex recovery failed: {e}")

        logger.error(f"Failed to parse JSON: {content[:100]}...")
        return {}

# Apply the patch
base_vlm.OpenAIVLM = MultiPortOpenAIVLM
print("Applied OpenAIVLM Monkey Patch for Load Balancing and Robust Parsing.")

# --- Configure for HumanOmni (Local vLLM Service) ---
Config.MODEL_PROVIDER = "openai"

# Define the pool of Base URLs (Ports 8001-8008)
PORTS = range(8104, 8107)
Config.OPENAI_BASE_URL = [f"http://localhost:{port}/v1" for port in PORTS]
Config.OPENAI_API_KEY = "EMPTY"
Config.MODEL_NAME = "HumanOmni-7B-Omni" 

# Disabled Multi-Model mode
Config.MODELS = []

# Enable video input
Config.USE_VIDEO_INPUT = True

# Dataset path
Config.DATA_ROOT_DIR = os.path.join(project_dir, "data_new")

# Output directory
Config.OUTPUT_DIR = "results/HumanOmni-7B-Omni"
Config.SAVE_LOG = True

# Workers configuration
Config.NUM_WORKERS = 8
Config.EVAL_NUM_WORKERS = 8

print(f"Starting HumanOmni Evaluation with Config:")
print(f"Model: {Config.MODEL_NAME}")
print(f"URLs: {len(Config.OPENAI_BASE_URL)} endpoints")
print(f"Data: {Config.DATA_ROOT_DIR}")
print("-" * 50)

import main

if __name__ == "__main__":
    main.main()
