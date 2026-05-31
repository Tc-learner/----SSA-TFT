"""数据层 —— TE过程数据加载、SSA分解、序列构造"""

from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD


# ============================================================
# 数据加载
# ============================================================

def load_te_mat(path: str) -> pd.DataFrame:
    """加载 TE 过程 .mat 文件，返回 simout 数据的 DataFrame"""
    try:
        data = loadmat(path)
    except Exception as e:
        raise IOError(f"无法加载 MAT 文件 {path}: {e}")

    if 'simout' in data:
        return pd.DataFrame(data['simout'])
    # 某些 .mat 文件用不同变量名
    for key in data:
        if not key.startswith('__'):
            arr = data[key]
            if arr.ndim >= 2 and arr.shape[1] > 1:
                return pd.DataFrame(arr)
    raise KeyError(f"未找到可用的数据变量，文件包含: {list(data.keys())}")


def load_ssa_result(path: str) -> np.ndarray:
    """加载 SSA 预处理后的结果文件（normalizeResult.mat），返回 n_src 数组"""
    data = loadmat(path)
    results = np.array(data['ssa_results'])
    return results['n_src'][0][0]


# ============================================================
# SSA 分解（奇异谱分析）
# ============================================================

def decompose_ssa(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    window_size: int = 12,
) -> pd.DataFrame:
    """
    对时序数据进行 SSA 分解

    Args:
        data: 一维或二维时序数据
        window_size: 滞后窗口大小

    Returns:
        components_df: 包含各主成分的 DataFrame
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = pd.Series(data)
        else:
            data = pd.DataFrame(data)

    # 构建滞后矩阵
    lagged_data = pd.concat(
        [data.shift(i) for i in range(window_size)], axis=1
    ).dropna()

    # 截断 SVD 分解
    svd = TruncatedSVD(n_components=window_size - 1)
    svd.fit(lagged_data)
    components = svd.transform(lagged_data)

    # 整理成分 DataFrame
    component_names = [f'component_{i}' for i in range(1, window_size)]
    components_df = pd.DataFrame(
        components,
        index=data.index[window_size - 1:],
        columns=component_names
    )
    return components_df


def multi_scale_ssa_decompose(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    window_sizes: List[int] = (6, 12, 24),
) -> pd.DataFrame:
    """多尺度 SSA 分解：在不同窗口大小下运行 SSA，拼接所有分量。

    Args:
        data: 一维或二维时序数据
        window_sizes: 多个滞后窗口大小列表

    Returns:
        拼接后的 DataFrame，列数为 sum(w-1 for w in window_sizes)
    """
    window_sizes = list(window_sizes)

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
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
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

    common_idx = comps_small.index.intersection(comps_large.index)
    recon_small = comps_small.loc[common_idx].sum(axis=1).values
    recon_large = comps_large.loc[common_idx].sum(axis=1).values

    return np.abs(recon_small - recon_large)


# ============================================================
# 序列构造与预处理
# ============================================================

def split_sequences(
    data: np.ndarray,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    滑动窗口构造输入/输出序列对

    Args:
        data: (n_samples, n_features) 数组
        n_steps: 输入序列长度

    Returns:
        X: (n_samples - n_steps, n_steps, n_features)
        y: (n_samples - n_steps, n_features)
    """
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        X.append(data[i:end_ix])
        y.append(data[end_ix])
    return np.array(X), np.array(y)


def padded_split_sequences(
    data: np.ndarray,
    n_steps: int,
    pad_data: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """滑动窗口构造序列，短片段时用 pad_data 填充。

    当 data 长度 < n_steps 时，用 pad_data 的前几个样本做填充，
    使得短片段也能构造出有效的预测序列。填充后的序列仅用于预测，
    标签仍来自原始 data。

    Args:
        data: (n_samples, n_features) 目标片段数据
        n_steps: 输入序列长度
        pad_data: 填充用数据（通常为下一片段的前 n_steps 个样本）。若为 None 且 data 不足，返回空数组。

    Returns:
        X: (n_samples_out, n_steps, n_features)
        y: (n_samples_out, n_features)
    """
    if len(data) >= n_steps:
        X, y = [], []
        for i in range(len(data)):
            end_ix = i + n_steps
            if end_ix > len(data) - 1:
                break
            X.append(data[i:end_ix])
            y.append(data[end_ix])
        return np.array(X) if X else np.empty((0, n_steps, data.shape[1])), \
               np.array(y) if y else np.empty((0, data.shape[1]))

    # data 太短，需要填充
    if pad_data is None or len(pad_data) == 0:
        return np.empty((0, n_steps, data.shape[1])), np.empty((0, data.shape[1]))

    pad_needed = n_steps - len(data) + 1
    available = min(pad_needed, len(pad_data))

    X, y = [], []
    for i in range(len(data)):
        seq_end = i + n_steps
        if seq_end >= len(data):
            # 需要填充
            from_data = data[i:len(data)]
            from_pad = pad_data[:seq_end - len(data)]
            seq = np.concatenate([from_data, from_pad], axis=0)
            if len(seq) >= n_steps:
                seq = seq[:n_steps]
        else:
            seq = data[i:seq_end]

        if len(seq) < n_steps:
            break

        target_idx = min(i + n_steps, len(data) - 1)
        X.append(seq)
        y.append(data[target_idx])

    return np.array(X) if X else np.empty((0, n_steps, data.shape[1])), \
           np.array(y) if y else np.empty((0, data.shape[1]))


def standardize(
    train_data: np.ndarray,
    test_data: Optional[np.ndarray] = None,
) -> Union[Tuple[StandardScaler, np.ndarray],
           Tuple[StandardScaler, np.ndarray, np.ndarray]]:
    """
    标准化数据（训练集 fit，测试集 transform）

    Returns:
        scaler, train_scaled, test_scaled (或只返回 scaler, train_scaled)
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    if test_data is not None:
        test_scaled = scaler.transform(test_data)
        return scaler, train_scaled, test_scaled
    return scaler, train_scaled


def split_train_test(
    data: np.ndarray,
    ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """按比例划分训练集和测试集"""
    split_idx = int(len(data) * ratio)
    return data[:split_idx], data[split_idx:]
