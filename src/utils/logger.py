# src/utils/logger.py
import logging
import os
import sys
from datetime import datetime


def setup_logger(output_dir=None, name="VSIG_Logger"):
    """
    配置并返回一个 logger 实例。

    Args:
        output_dir (str, optional): 日志文件保存目录。如果不提供，则只输出到控制台。
        name (str): logger 名称。

    Returns:
        logging.Logger: 配置好的 logger 对象。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 格式化
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 文件 Handler (如果指定了目录)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"run_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        print(f"日志将保存至: {log_file}")

    return logger
