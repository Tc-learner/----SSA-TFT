"""短过渡检测模块 —— 多尺度 SSA、双向确认、变化点预筛选"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

from .data import decompose_ssa, split_sequences


# ============================================================
# 多尺度 SSA
# ============================================================

def multi_scale_ssa_decompose(
    data: np.ndarray,
    window_sizes: List[int] = (6, 12, 24),
) -> pd.DataFrame:
    """多尺度 SSA 分解：在不同窗口大小下运行 SSA，拼接所有分量。

    Args:
        data: 一维或二维时序数据
        window_sizes: 多个滞后窗口大小列表

    Returns:
        拼接后的 DataFrame，列数为 sum(w-1 for w in window_sizes)
    """
    if isinstance(window_sizes, (list, tuple)):
        window_sizes = list(window_sizes)
    else:
        window_sizes = [window_sizes]

    component_list = []
    for ws in window_sizes:
        comps = decompose_ssa(data, window_size=ws)
        comps.columns = [f'w{ws}_{c}' for c in comps.columns]
        component_list.append(comps)

    # 对齐索引（取最小窗口的索引范围）
    common_index = component_list[0].index
    for comps in component_list[1:]:
        common_index = common_index.intersection(comps.index)

    aligned = [comps.loc[common_index] for comps in component_list]
    return pd.concat(aligned, axis=1)


def ssa_residual_feature(
    data: np.ndarray,
    window_small: int = 6,
    window_large: int = 12,
) -> np.ndarray:
    """计算细粒度与粗粒度 SSA 重建之间的残差特征。

    残差在短过渡（高频变化）时出现尖峰，可作为辅助检测特征。

    Args:
        data: 一维或二维时序数据
        window_small: 细粒度窗口大小
        window_large: 粗粒度窗口大小

    Returns:
        残差序列 (n_samples,)，每个值 >= 0
    """
    comps_small = decompose_ssa(data, window_size=window_small)
    comps_large = decompose_ssa(data, window_size=window_large)

    # 对齐索引
    common_idx = comps_small.index.intersection(comps_large.index)
    recon_small = comps_small.loc[common_idx].sum(axis=1).values
    recon_large = comps_large.loc[common_idx].sum(axis=1).values

    return np.abs(recon_small - recon_large)


# ============================================================
# 双向残差确认
# ============================================================

def bidirectional_confirm(
    model_class,
    X_all: np.ndarray,
    y_all: np.ndarray,
    trigger_pos: int,
    threshold: float,
    n_steps: int,
    lookback: int = 10,
    lookahead: int = 25,
    fast_epochs: int = 10,
    batch_size: int = 32,
) -> bool:
    """双向残差确认：从触发点两侧交叉验证模态变化是否真实。

    原理：
    - 前向：触发点之前的模型预测触发点之后的数据，RMSE 应偏高
    - 后向：用触发点之后的少量数据快速训练一个新模型，回测之前的数据，RMSE 也应偏高
    - 真正的模态边界从两侧看都应异常

    Args:
        model_class: create_model 函数引用
        X_all: 全部输入序列
        y_all: 全部标签
        trigger_pos: 候选触发点位置
        threshold: RMSE 阈值
        n_steps: LSTM 输入步长
        lookback: 回测点数
        lookahead: 前看点数和后向训练窗口大小
        fast_epochs: 后向模型的快速训练轮数
        batch_size: 批次大小

    Returns:
        True 如果双向都确认异常
    """
    total = len(X_all)

    # 前向检查：触发点之后的点是否仍超阈值
    forward_end = min(trigger_pos + 5, total)
    forward_rmse_list = []
    for i in range(trigger_pos, forward_end):
        X_cur = X_all[i:i + 1]
        y_cur = y_all[i:i + 1]
        y_pred = model_class.predict(X_cur, verbose=0)
        from .trainer import compute_rmse
        forward_rmse_list.append(compute_rmse(y_cur[0], y_pred[0]))

    forward_mean = np.mean(forward_rmse_list)

    # 后向检查：用触发点之后的数据快速训练一个模型
    post_start = min(trigger_pos + 5, total - lookahead)
    post_end = min(post_start + lookahead, total)
    if post_end - post_start < n_steps + 3:
        # 后续数据不足，仅依赖前向结果
        return forward_mean > threshold

    X_post, y_post = split_sequences(
        y_all[post_start:post_end], n_steps
    )
    if len(X_post) == 0:
        return forward_mean > threshold

    input_shape = (n_steps, X_post.shape[2])
    n_features = X_post.shape[2]

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    # 用相同架构快速训练后向模型
    post_model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(n_features),
    ])
    post_model.compile(optimizer='adam', loss='mse')
    post_model.fit(X_post, y_post, epochs=fast_epochs,
                   batch_size=batch_size, verbose=0)

    # 后向预测：用后向模型预测触发点之前的数据
    backward_start = max(n_steps, trigger_pos - lookback)
    backward_rmse_list = []
    for i in range(backward_start, trigger_pos):
        X_cur = X_all[i:i + 1]
        y_cur = y_all[i:i + 1]
        y_pred = post_model.predict(X_cur, verbose=0)
        from .trainer import compute_rmse
        backward_rmse_list.append(compute_rmse(y_cur[0], y_pred[0]))

    backward_mean = np.mean(backward_rmse_list) if backward_rmse_list else 0

    return forward_mean > threshold and backward_mean > threshold


# ============================================================
# 变化点预筛选（基于 ruptures 库）
# ============================================================

def change_point_prescreen(
    y_all: np.ndarray,
    min_size: int = 10,
    penalty: float = 5.0,
) -> List[int]:
    """使用 PELT 算法在全序列上做变化点粗筛。

    Args:
        y_all: (n_samples, n_features) 标签数据
        min_size: 最小段长度
        penalty: PELT 惩罚系数（越大检测越少）

    Returns:
        候选变化点位置列表
    """
    try:
        import ruptures as rpt
    except ImportError:
        raise ImportError("ruptures 库未安装，请运行: pip install ruptures")

    # 在多变量信号上取第一主成分或均值作为 PELT 输入
    if y_all.ndim == 2 and y_all.shape[1] > 1:
        signal = np.mean(y_all, axis=1)
    else:
        signal = y_all.ravel() if y_all.ndim > 1 else y_all

    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
    breakpoints = algo.predict(pen=penalty)

    return breakpoints[:-1]  # 去掉末尾的数据终点


# ============================================================
# 统计距离回退（极短片段分类）
# ============================================================

def mahalanobis_distance(
    segment: np.ndarray,
    ref_mean: np.ndarray,
    ref_cov_inv: np.ndarray,
) -> float:
    """计算 segment 均值向量与参考分布之间的马氏距离。

    Args:
        segment: (n_samples, n_features) 的片段数据
        ref_mean: 参考分布的均值向量
        ref_cov_inv: 参考分布协方差矩阵的逆

    Returns:
        马氏距离标量
    """
    seg_mean = np.mean(segment, axis=0)
    diff = seg_mean - ref_mean
    dist = np.sqrt(np.dot(np.dot(diff.T, ref_cov_inv), diff))
    return float(dist)


def classify_short_segment(
    segment: np.ndarray,
    prev_mode_mean: np.ndarray,
    prev_mode_cov: np.ndarray,
    threshold: float = 3.0,
) -> dict:
    """对过短的片段做统计分类（无法使用 LSTM 时回退）。

    Args:
        segment: 短片段数据 (n_samples, n_features)
        prev_mode_mean: 前一已知模态的均值向量
        prev_mode_cov: 前一已知模态的协方差矩阵
        threshold: 3-sigma 规则的马氏距离阈值

    Returns:
        {
            'type': 'short_transient',
            'length': 片段长度,
            'mahalanobis_distance': 马氏距离,
            'is_transition': 是否认定为过渡
        }
    """
    n_features = prev_mode_mean.shape[0]
    # 正则化协方差求逆
    reg_cov = prev_mode_cov + np.eye(n_features) * 1e-6
    try:
        cov_inv = np.linalg.inv(reg_cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(reg_cov)

    dist = mahalanobis_distance(segment, prev_mode_mean, cov_inv)

    return {
        'type': 'short_transient',
        'length': len(segment),
        'mahalanobis_distance': dist,
        'is_transition': dist > threshold,
    }
