"""实验脚本 —— 模型对比"""

import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config, DEFAULT_DATA_FILE
from src.data import load_te_mat, decompose_ssa, split_sequences, \
    standardize, split_train_test
from src.models import create_model
from src.trainer import compute_rmse
from src.utils import set_global_determinism
from src.logger import get_logger
from src.visualization import plot_model_compare
from src.evaluation import save_evaluation_report

logger = get_logger(__name__)


def run_single_model(model_type: str, X_train: np.ndarray,
                     y_train: np.ndarray, X_test: np.ndarray,
                     y_test: np.ndarray, config: dict) -> np.ndarray:
    """训练并评估单个模型，返回 RMSE 列表（TFT 自动提取 P50）"""
    input_shape = (config['N_STEPS'], X_train.shape[2])
    units = config['TFT_UNITS'] if model_type == 'tft' else config['LSTM_UNITS']
    model = create_model(model_type, input_shape, units)

    model.fit(X_train, y_train, epochs=config['EPOCHS'],
              batch_size=config['BATCH_SIZE'], verbose=0)

    # 局部导入 TFT 预测提取函数
    if model_type == 'tft':
        from src.trainer import _get_p50_prediction
        n_quantiles = len(config.get('TFT_QUANTILES', [0.1, 0.5, 0.9]))

    rmse_list = []
    for i in range(len(X_test)):
        X = X_test[i:i + 1]
        y = y_test[i:i + 1]
        y_pred_raw = model.predict(X, verbose=0)
        if model_type == 'tft':
            y_pred = _get_p50_prediction(model, y_pred_raw[0:1], config)[0:1]
        else:
            y_pred = y_pred_raw[0:1]
        rmse = compute_rmse(y[0], y_pred[0])
        rmse_list.append(rmse)

    return np.array(rmse_list)


def run(config: dict) -> dict:
    set_global_determinism(seed=config['SEED'])

    data_path = config.get('DEFAULT_DATA_FILE', DEFAULT_DATA_FILE)
    logger.info(f"加载数据: {data_path}")
    df = load_te_mat(data_path)

    # SSA 分解
    components = decompose_ssa(df, window_size=config['WINDOW_SIZE'])

    # 划分训练/测试
    train_raw, test_raw = split_train_test(components.values,
                                           ratio=config['TRAIN_RATIO'])

    # 标准化
    scaler, train_scaled, test_scaled = standardize(train_raw, test_raw)

    # 构造序列
    X_train, y_train = split_sequences(train_scaled, config['N_STEPS'])
    X_test, y_test = split_sequences(test_scaled, config['N_STEPS'])

    logger.info(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")

    models_to_compare = [
        ("tft", "SSA-TFT (核心方法)"),
        ("lstm", "标准 LSTM"),
        ("attention_lstm", "Attention LSTM"),
        ("robust_lstm", "Robust LSTM"),
        ("control_gate_lstm", "Control-Gate LSTM"),
    ]

    results = {}
    result_dir = config['RESULT_DIR']
    os.makedirs(result_dir, exist_ok=True)

    for model_type, label in models_to_compare:
        logger.info(f"训练 {label} ...")
        try:
            rmse_arr = run_single_model(model_type, X_train, y_train,
                                        X_test, y_test, config)

            csv_path = os.path.join(result_dir, f'{model_type}_rmse.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                for r in rmse_arr:
                    writer.writerow([r])

            results[label] = rmse_arr
            logger.info(f"  {label}: 平均 RMSE = {rmse_arr.mean():.4f}")
        except Exception as e:
            logger.error(f"  {label} 失败: {e}")

    # 绘制对比图
    try:
        plot_model_compare(
            results,
            title=f"模型对比: RMSE ({os.path.basename(data_path)})",
            save_path=os.path.join(result_dir, 'compare_rmse.png'),
        )
    except Exception as e:
        logger.error(f"绘图失败: {e}")

    # 打印汇总
    print("\n====== 对比结果汇总 ======")
    for label, rmse_arr in results.items():
        print(f"{label}: 均值 = {rmse_arr.mean():.4f}, "
              f"标准差 = {rmse_arr.std():.4f}")

    return results


if __name__ == "__main__":
    config = get_config()
    config['DEFAULT_DATA_FILE'] = DEFAULT_DATA_FILE
    config['EPOCHS'] = 50
    run(config)
