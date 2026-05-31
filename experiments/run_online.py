"""实验脚本 —— 在线模态变化检测"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config, DEFAULT_DATA_FILE
from src.data import load_te_mat, decompose_ssa, split_sequences, standardize, \
    multi_scale_ssa_decompose
from src.trainer import OnlineDetector
from src.utils import set_global_determinism, detect_gpu
from src.logger import get_logger
from src.evaluation import compute_detection_metrics, save_evaluation_report
from src.visualization import plot_rmse_stream

logger = get_logger(__name__)


def run(config: dict) -> list:
    set_global_determinism(seed=config['SEED'])

    data_path = config.get('DEFAULT_DATA_FILE', DEFAULT_DATA_FILE)
    logger.info(f"加载数据: {data_path}")
    df = load_te_mat(data_path)

    # SSA 分解（支持多尺度）
    if config.get('ENABLE_MULTI_SCALE_SSA', False):
        window_sizes = config.get('SSA_WINDOW_SIZES', [6, 12, 24])
        logger.info(f"SSA 多尺度窗口 = {window_sizes}")
        components = multi_scale_ssa_decompose(df, window_sizes=window_sizes)
    else:
        logger.info(f"SSA 窗口大小 = {config['WINDOW_SIZE']}")
        components = decompose_ssa(df, window_size=config['WINDOW_SIZE'])

    # 标准化
    scaler, scaled_data = standardize(components.values)
    # 构造序列
    X_all, y_all = split_sequences(scaled_data, config['N_STEPS'])

    logger.info(f"总样本数: {len(X_all)}, 输入形状: {X_all.shape[1:]}, "
                f"输出维度: {y_all.shape[1]}")

    # GPU
    gpu_ok, gpu_dev = detect_gpu()

    # 运行在线检测
    detector = OnlineDetector(config)
    transitions = detector.run(X_all, y_all,
                               gpu_device=gpu_dev if gpu_ok else None)

    logger.info(f"检测结果: 共发现 {len(transitions)} 次模态变化")
    for t in transitions:
        logger.info(f"  样本位置: {t}")

    # 评估（如有地面真值）
    try:
        import yaml
        gt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'configs', 'ground_truth.yaml')
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                gt_data = yaml.safe_load(f)
            fname = os.path.basename(data_path)
            if fname in gt_data and gt_data[fname].get('transitions'):
                gt_transitions = gt_data[fname]['transitions']
                metrics = compute_detection_metrics(
                    transitions, gt_transitions, total_length=len(X_all)
                )
                logger.info(f"  F1: {metrics['f1']:.4f}, "
                            f"均值延迟: {metrics['mean_delay']}")
                save_evaluation_report(metrics, config['RESULT_DIR'],
                                       prefix='online_eval')
    except Exception:
        pass

    # 可视化
    try:
        import pandas as pd
        loss_df = pd.read_csv(detector.loss_csv, header=None)
        rmse_stream = loss_df.values.flatten()
        plot_rmse_stream(
            rmse_stream, detector.threshold, transitions,
            title=f"在线检测 RMSE - {os.path.basename(data_path)}",
            save_path=os.path.join(config['RESULT_DIR'], 'online_rmse.png'),
        )
    except Exception:
        pass

    return transitions


if __name__ == "__main__":
    config = get_config()
    config['DEFAULT_DATA_FILE'] = DEFAULT_DATA_FILE
    # MODEL_TYPE 默认为 'tft'（核心方法），可通过 YAML 覆盖
    run(config)
