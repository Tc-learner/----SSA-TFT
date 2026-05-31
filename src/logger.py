"""日志模块 —— 结构化日志，统一替代 print()"""

import logging
import sys
import os
from typing import Optional


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """获取配置好的 Logger 实例。

    Args:
        name: 日志器名称（通常使用 __name__）
        log_file: 日志文件路径（可选，None 则仅输出到控制台）
        level: 日志级别

    Returns:
        配置完成的 Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        # 控制台输出
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # 文件输出
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
