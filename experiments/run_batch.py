"""批量实验运行器 —— 参数扫描、交叉验证、模型对比"""

import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config, validate_config, DEFAULT_DATA_FILE, RESULT_DIR
from src.data import load_te_mat, decompose_ssa, split_sequences, standardize
from src.trainer import OnlineDetector
from src.evaluation import (compute_detection_metrics,
                            threshold_sensitivity_curve,
                            compute_auc,
                            save_evaluation_report)
from src.experiment_tracker import ExperimentTracker


def run_threshold_sweep(
    config: dict,
    multipliers: np.ndarray = None,
) -> dict:
    """扫描 THRESHOLD_MULTIPLIER 做敏感性分析。

    Args:
        config: 基础配置
        multipliers: 要扫描的乘数序列 (默认 1.0 到 5.0)

    Returns:
        {'thresholds': [...], 'tpr': [...], 'fpr': [...], 'auc': ...}
    """
    if multipliers is None:
        multipliers = np.arange(1.0, 5.0, 0.25)

    tracker = ExperimentTracker("threshold_sweep", RESULT_DIR)
    tracker.log_params({
        **{k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))},
        'multipliers': multipliers.tolist(),
    })

    # 加载数据
    data_path = config.get('DEFAULT_DATA_FILE', DEFAULT_DATA_FILE)
    print(f"[批量] 加载: {data_path}")
    df = load_te_mat(data_path)
    components = decompose_ssa(df, window_size=config['WINDOW_SIZE'])
    scaler, scaled_data = standardize(components.values)
    X_all, y_all = split_sequences(scaled_data, config['N_STEPS'])

    all_rmse_streams = {}

    for mult in multipliers:
        cfg = config.copy()
        cfg['THRESHOLD_MULTIPLIER'] = float(mult)

        detector = OnlineDetector(cfg)
        transitions = detector.run(X_all, y_all)

        # 统计预测过渡点
        predicted = list(transitions)
        metrics = compute_detection_metrics(
            predicted,
            config.get('true_transitions', []),
            total_length=len(X_all),
        )

        tracker.log_metrics({
            'threshold_multiplier': float(mult),
            'n_detected': len(transitions),
            **{k: v for k, v in metrics.items() if not isinstance(v, list)},
        }, step=int(mult * 100))

    tracker.save_summary()
    print(f"[批量] 阈值扫描完成，结果: {tracker.run_dir}")
    return tracker.metrics


def run_data_cross_validation(
    config: dict,
    data_dir: str = None,
) -> dict:
    """在所有 MAT 文件上运行同一流程。

    Args:
        config: 基础配置
        data_dir: 数据目录（默认使用 config 中的路径）

    Returns:
        {filename: {'transitions': [...], 'metrics': {...}}, ...}
    """
    if data_dir is None:
        data_dir = os.path.dirname(config.get('DEFAULT_DATA_FILE',
                                              DEFAULT_DATA_FILE))

    mat_files = sorted([f for f in os.listdir(data_dir)
                        if f.endswith('.mat') and not f.startswith('~')])

    tracker = ExperimentTracker("cross_validation", RESULT_DIR)
    results = {}

    for mat_file in mat_files:
        data_path = os.path.join(data_dir, mat_file)
        print(f"\n[批量] 处理: {mat_file}")

        try:
            df = load_te_mat(data_path)
            components = decompose_ssa(df, window_size=config['WINDOW_SIZE'])
            scaler, scaled_data = standardize(components.values)
            X_all, y_all = split_sequences(scaled_data, config['N_STEPS'])

            cfg = config.copy()
            cfg['DEFAULT_DATA_FILE'] = data_path
            detector = OnlineDetector(cfg)
            transitions = detector.run(X_all, y_all)

            metrics = compute_detection_metrics(
                list(transitions),
                config.get('true_transitions', []),
                total_length=len(X_all),
            )

            results[mat_file] = {
                'transitions': list(transitions),
                'metrics': metrics,
            }

            tracker.log_metrics({
                'file': mat_file,
                'n_transitions': len(transitions),
                **{k: v for k, v in metrics.items() if not isinstance(v, list)},
            })

        except Exception as e:
            print(f"[错误] {mat_file}: {e}")
            results[mat_file] = {'error': str(e)}
            continue

    tracker.save_summary()

    # 保存汇总 CSV
    summary_path = os.path.join(tracker.run_dir, 'cross_validation_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'n_transitions', 'precision', 'recall', 'f1',
                         'mean_delay'])
        for mat_file, res in results.items():
            if 'error' in res:
                writer.writerow([mat_file, 'ERROR', '', '', '', ''])
            else:
                m = res['metrics']
                writer.writerow([
                    mat_file, len(res['transitions']),
                    m['precision'], m['recall'], m['f1'],
                    m['mean_delay'],
                ])

    print(f"\n[批量] 交叉验证完成: {summary_path}")
    return results


def run_model_comparison_full(config: dict) -> dict:
    """在所有模型类型上运行对比实验。

    Returns:
        {model_type: {'mean_rmse': ..., 'std_rmse': ..., ...}, ...}
    """
    from src.models import create_model
    from src.trainer import compute_rmse
    from src.data import split_train_test

    model_types = ["lstm", "attention_lstm", "robust_lstm",
                   "control_gate_lstm", "tft"]

    data_path = config.get('DEFAULT_DATA_FILE', DEFAULT_DATA_FILE)
    print(f"[对比] 加载: {data_path}")
    df = load_te_mat(data_path)
    components = decompose_ssa(df, window_size=config['WINDOW_SIZE'])

    train_raw, test_raw = split_train_test(components.values,
                                           ratio=config['TRAIN_RATIO'])
    scaler, train_scaled, test_scaled = standardize(train_raw, test_raw)
    X_train, y_train = split_sequences(train_scaled, config['N_STEPS'])
    X_test, y_test = split_sequences(test_scaled, config['N_STEPS'])

    tracker = ExperimentTracker("model_comparison", RESULT_DIR)
    tracker.log_params({k: v for k, v in config.items()
                        if isinstance(v, (int, float, str, bool))})

    results = {}
    input_shape = (config['N_STEPS'], X_train.shape[2])

    for model_type in model_types:
        print(f"[对比] 训练 {model_type} ...")
        units = config['TFT_UNITS'] if model_type == 'tft' else config['LSTM_UNITS']

        try:
            model = create_model(model_type, input_shape, units)
            model.fit(X_train, y_train, epochs=config['EPOCHS'],
                      batch_size=config['BATCH_SIZE'], verbose=0)

            rmse_list = []
            for i in range(len(X_test)):
                y_pred = model.predict(X_test[i:i + 1], verbose=0)
                rmse = compute_rmse(y_test[i], y_pred[0])
                rmse_list.append(rmse)

            rmse_arr = np.array(rmse_list)
            results[model_type] = {
                'mean_rmse': float(np.mean(rmse_arr)),
                'std_rmse': float(np.std(rmse_arr)),
                'min_rmse': float(np.min(rmse_arr)),
                'max_rmse': float(np.max(rmse_arr)),
            }

            tracker.log_metrics({
                'model_type': model_type,
                **results[model_type],
            })

            # 保存逐点 RMSE
            csv_path = os.path.join(tracker.run_dir, f'{model_type}_rmse.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                for r in rmse_arr:
                    writer.writerow([r])

            print(f"  {model_type}: mean_rmse={results[model_type]['mean_rmse']:.4f}")

        except Exception as e:
            print(f"[错误] {model_type}: {e}")
            results[model_type] = {'error': str(e)}

    tracker.save_summary()
    print(f"\n[对比] 完成，结果: {tracker.run_dir}")
    return results
