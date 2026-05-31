"""实验脚本 —— 标准 SSA+TFT 训练与评估"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config, DEFAULT_DATA_FILE
from src.trainer import run_standard_pipeline
from src.utils import set_global_determinism
from src.logger import get_logger

logger = get_logger(__name__)


def run(config: dict) -> float:
    set_global_determinism(seed=config['SEED'])
    if 'DEFAULT_DATA_FILE' not in config:
        config['DEFAULT_DATA_FILE'] = DEFAULT_DATA_FILE
    rmse = run_standard_pipeline(config)
    logger.info(f"最终 RMSE: {rmse:.4f}")
    return rmse


if __name__ == "__main__":
    config = get_config()
    config['DEFAULT_DATA_FILE'] = DEFAULT_DATA_FILE
    config['MODEL_TYPE'] = 'tft'
    config['FULL_EPOCHS'] = 200
    run(config)
