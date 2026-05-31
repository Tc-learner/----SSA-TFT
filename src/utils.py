"""工具函数 —— 随机种子设置与 GPU 检测"""

import os
import random
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf


def set_seeds(seed: int = 11) -> None:
    """设置 Python、NumPy、TensorFlow 的随机种子"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_global_determinism(seed: int = 11) -> None:
    """全面设置确定性（确保实验结果可复现）"""
    set_seeds(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def detect_gpu() -> Tuple[bool, Optional[str]]:
    """检测并配置 GPU，返回 (是否可用, 设备字符串)"""
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        device = '/' + ':'.join(str(x) for x in physical_devices[0].name.split(':')[1:3])
        return True, device
    return False, None
