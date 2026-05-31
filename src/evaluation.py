"""评估模块 —— 模态变化检测的精度、延迟、敏感性分析"""

import os
import numpy as np
import csv
from typing import List, Tuple, Dict, Optional


# ============================================================
# 地面真值匹配
# ============================================================

def match_transitions(
    predicted: List[int],
    ground_truth: List[int],
    tolerance: int = 5,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """将预测的过渡点与地面真值做容差匹配。

    Args:
        predicted: 检测到的过渡点位置列表
        ground_truth: 真实的过渡点位置列表
        tolerance: 容差窗口大小（样本数）

    Returns:
        (matched_pairs, unmatched_pred, unmatched_gt)
        - matched_pairs: [(pred_pos, gt_pos), ...]
        - unmatched_pred: 未能匹配的预测过渡点
        - unmatched_gt: 未能匹配的真实过渡点
    """
    matched = []
    used_pred = set()
    used_gt = set()

    for pi, pred_pos in enumerate(predicted):
        best_dist = tolerance + 1
        best_gi = -1
        for gi, gt_pos in enumerate(ground_truth):
            if gi in used_gt:
                continue
            dist = abs(pred_pos - gt_pos)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_gi = gi
        if best_gi >= 0:
            matched.append((pred_pos, ground_truth[best_gi]))
            used_pred.add(pi)
            used_gt.add(best_gi)

    unmatched_pred = [p for i, p in enumerate(predicted) if i not in used_pred]
    unmatched_gt = [g for i, g in enumerate(ground_truth) if i not in used_gt]

    return matched, unmatched_pred, unmatched_gt


# ============================================================
# 检测精度指标
# ============================================================

def compute_detection_metrics(
    predicted: List[int],
    ground_truth: List[int],
    total_length: int,
    tolerance: int = 5,
) -> dict:
    """计算模态变化检测的完整评估指标。

    Args:
        predicted: 检测到的过渡点位置
        ground_truth: 真实过渡点位置
        total_length: 总样本数
        tolerance: 匹配容差窗口

    Returns:
        {
            'precision': ..., 'recall': ..., 'f1': ...,
            'tp': ..., 'fp': ..., 'fn': ...,
            'false_positive_rate_per_1000': ...,
            'detection_delays': [...],
            'mean_delay': ..., 'median_delay': ..., 'max_delay': ...,
        }
    """
    matched, unmatched_pred, unmatched_gt = match_transitions(
        predicted, ground_truth, tolerance
    )

    tp = len(matched)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # 检测延迟
    delays = [abs(pred - gt) for pred, gt in matched]

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'false_positive_rate_per_1000': fp / (total_length / 1000) if total_length > 0 else 0.0,
        'detection_delays': delays,
        'mean_delay': float(np.mean(delays)) if delays else None,
        'median_delay': float(np.median(delays)) if delays else None,
        'max_delay': int(np.max(delays)) if delays else None,
    }


# ============================================================
# 阈值敏感性 AUC
# ============================================================

def threshold_sensitivity_curve(
    rmse_stream: np.ndarray,
    ground_truth: List[int],
    total_length: int,
    num_thresholds: int = 50,
    tolerance: int = 5,
) -> Dict[str, np.ndarray]:
    """扫描一系列阈值，生成 TPR/FPR 曲线数据。

    Args:
        rmse_stream: RMSE 序列 (n_samples,)
        ground_truth: 真实过渡位置
        total_length: 总样本数
        num_thresholds: 扫描点数
        tolerance: 匹配容差

    Returns:
        {'thresholds': [...], 'tpr': [...], 'fpr': [...], 'f1': [...]}
    """
    if len(rmse_stream) == 0:
        return {'thresholds': np.array([]), 'tpr': np.array([]),
                'fpr': np.array([]), 'f1': np.array([])}

    min_t = np.min(rmse_stream)
    max_t = np.max(rmse_stream)
    thresholds = np.linspace(min_t, max_t, num_thresholds)

    tpr_list, fpr_list, f1_list = [], [], []

    for th in thresholds:
        # 简单检测：所有超阈值点作为预测
        predicted = []
        in_event = False
        for i, r in enumerate(rmse_stream):
            if r > th:
                if not in_event:
                    predicted.append(i)
                    in_event = True
            else:
                in_event = False

        metrics = compute_detection_metrics(
            predicted, ground_truth, total_length, tolerance
        )
        tpr_list.append(metrics['recall'])
        fpr_list.append(metrics['false_positive_rate_per_1000'])
        f1_list.append(metrics['f1'])

    return {
        'thresholds': thresholds,
        'tpr': np.array(tpr_list),
        'fpr': np.array(fpr_list),
        'f1': np.array(f1_list),
    }


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """计算 AUC (trapezoidal rule)"""
    if len(fpr) < 2:
        return 0.0
    # Sort by FPR
    idx = np.argsort(fpr)
    fpr_sorted = fpr[idx]
    tpr_sorted = tpr[idx]
    return float(np.trapz(tpr_sorted, fpr_sorted))


# ============================================================
# 污染分数
# ============================================================

def compute_contamination_score(
    incremental_rmse: np.ndarray,
    fresh_rmse: np.ndarray,
) -> dict:
    """比较增量微调模型与从头重训模型的性能。

    Args:
        incremental_rmse: 增量微调模型在后过渡数据上的逐点 RMSE
        fresh_rmse: 从头重训模型在相同数据上的逐点 RMSE

    Returns:
        {
            'incremental_mean': ...,
            'fresh_mean': ...,
            'degradation_ratio': ...,  # > 1 表示增量模型更差
            'is_contaminated': ...,    # 退化比超过 1.5 认为污染
        }
    """
    inc_mean = float(np.mean(incremental_rmse))
    fresh_mean = float(np.mean(fresh_rmse))
    degradation = inc_mean / fresh_mean if fresh_mean > 0 else 1.0

    return {
        'incremental_mean': inc_mean,
        'fresh_mean': fresh_mean,
        'degradation_ratio': degradation,
        'is_contaminated': degradation > 1.5,
    }


# ============================================================
# 逐模态 RMSE 稳定性
# ============================================================

def per_mode_stability(
    rmse_stream: np.ndarray,
    mode_boundaries: List[int],
) -> List[dict]:
    """计算每个模态段内的 RMSE 统计信息。

    Args:
        rmse_stream: 完整 RMSE 序列
        mode_boundaries: 模态边界位置列表（递增排列）

    Returns:
        [{'mode_id': ..., 'start': ..., 'end': ..., 'mean': ..., 'std': ..., 'n_points': ...}, ...]
    """
    boundaries = [0] + mode_boundaries + [len(rmse_stream)]
    results = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        segment = rmse_stream[start:end]
        if len(segment) == 0:
            continue
        results.append({
            'mode_id': i,
            'start': start,
            'end': end,
            'mean': float(np.mean(segment)),
            'std': float(np.std(segment)),
            'n_points': len(segment),
        })

    return results


# ============================================================
# 汇总报告
# ============================================================

def save_evaluation_report(
    metrics: dict,
    output_dir: str,
    prefix: str = "eval",
) -> str:
    """将评估结果保存为 CSV 汇总报告。

    Returns:
        报告文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 主指标
    summary_path = os.path.join(output_dir, f'{prefix}_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key in ['precision', 'recall', 'f1', 'tp', 'fp', 'fn',
                     'false_positive_rate_per_1000',
                     'mean_delay', 'median_delay', 'max_delay']:
            if key in metrics:
                writer.writerow([key, metrics[key]])

    # 延迟分布
    if 'detection_delays' in metrics and metrics['detection_delays']:
        delay_path = os.path.join(output_dir, f'{prefix}_delays.csv')
        with open(delay_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['delay'])
            for d in metrics['detection_delays']:
                writer.writerow([d])

    return summary_path
