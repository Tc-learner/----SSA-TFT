"""可视化模块 —— 统一的图表绘制函数"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.figure import Figure
from typing import Optional, List, Dict

_logger = logging.getLogger(__name__)


def setup_chinese_font() -> None:
    """配置 matplotlib 中文字体，按优先级尝试多个字体。

    检测系统可用的中文字体并设置到 rcParams，避免乱码。
    优先级：Noto Sans SC（思源黑体）> Microsoft YaHei > SimHei > SimSun。
    """
    _candidate_fonts = [
        'Noto Sans SC',        # Google 思源黑体（跨平台）
        'Noto Sans SC Medium',
        'Noto Serif SC',       # Google 思源宋体
        'WenQuanYi Micro Hei', # Linux 文泉驿
        'WenQuanYi Zen Hei',
        'Microsoft YaHei',     # Windows 微软雅黑
        'SimHei',              # Windows 黑体
        'SimSun',              # Windows 宋体
        'FangSong',            # Windows 仿宋
        'KaiTi',               # Windows 楷体
        'AR PL UMing CN',      # Linux
        'AR PL UKai CN',       # Linux
        'Noto Sans CJK SC',    # CJK 变体
        'Source Han Sans SC',  # Adobe 思源黑体
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    matched = [fn for fn in _candidate_fonts if fn in available]

    if matched:
        plt.rcParams['font.sans-serif'] = matched + ['sans-serif']
        plt.rcParams['font.family'] = 'sans-serif'
        _logger.debug(f'中文字体已配置: {matched[0]} (备选: {matched[1:]})')
    else:
        # 无匹配字体时发出警告，但不中断程序
        _logger.warning(
            '未检测到中文字体，图表中的中文可能显示为方框。'
            '请安装中文字体，如: Noto Sans SC, WenQuanYi Micro Hei, Microsoft YaHei。'
        )
        plt.rcParams['font.sans-serif'] = ['sans-serif']

    plt.rcParams['axes.unicode_minus'] = False


# 模块导入时自动配置中文字体
setup_chinese_font()


def _ensure_dir(save_path: Optional[str]) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)


# ============================================================
# RMSE 流图
# ============================================================

def plot_rmse_stream(
    rmse_values: np.ndarray,
    threshold: float,
    transitions: Optional[List[int]] = None,
    title: str = "RMSE 在线监测",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Figure:
    """绘制 RMSE 流及阈值线、检测到的过渡点。

    Args:
        rmse_values: RMSE 序列
        threshold: 检测阈值
        transitions: 检测到的过渡点位置
        title: 图表标题
        save_path: 保存路径（可选）
        dpi: 分辨率

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rmse_values, alpha=0.7, linewidth=0.8, label='RMSE')
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=1.5,
               label=f'阈值 = {threshold:.4f}')

    if transitions:
        y_min, y_max = ax.get_ylim()
        for pos in transitions:
            ax.axvline(x=pos, color='orange', linestyle=':', alpha=0.7)
            ax.text(pos, y_max * 0.95, str(pos), fontsize=8,
                    ha='center', color='orange')

    ax.set_xlabel('样本序号')
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    return fig


# ============================================================
# 检测延迟直方图
# ============================================================

def plot_detection_delay_histogram(
    delays: List[int],
    title: str = "检测延迟分布",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Figure:
    """绘制检测延迟的直方图。

    Args:
        delays: 各过渡点的检测延迟（样本数）
        title: 图表标题
        save_path: 保存路径
        dpi: 分辨率
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(delays, bins=max(5, len(delays) // 2),
            edgecolor='black', alpha=0.7)
    ax.axvline(x=np.median(delays), color='r', linestyle='--',
               label=f'中位数 = {np.median(delays):.1f}')
    ax.set_xlabel('检测延迟 (样本数)')
    ax.set_ylabel('频次')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    return fig


# ============================================================
# 阈值敏感性曲线
# ============================================================

def plot_threshold_sensitivity_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    title: str = "阈值敏感性 (ROC-like)",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Figure:
    """绘制 TPR vs FPR 的阈值敏感性曲线。

    Args:
        fpr: 假阳性率数组
        tpr: 真阳性率（召回率）数组
        auc: 曲线下面积
        title: 图表标题
        save_path: 保存路径
        dpi: 分辨率
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, 'b-', linewidth=1.5, label=f'AUC = {auc:.4f}')
    ax.plot([0, max(fpr)], [0, max(tpr)], 'k--', alpha=0.3)
    ax.set_xlabel('假阳性率 / 1000 样本')
    ax.set_ylabel('召回率 (TPR)')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    return fig


# ============================================================
# 模型对比图
# ============================================================

def plot_model_compare(
    results: Dict[str, np.ndarray],
    max_points: int = 200,
    title: str = "模型对比: RMSE 曲线",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Figure:
    """绘制多模型 RMSE 对比曲线。

    Args:
        results: {模型名: RMSE数组, ...}
        max_points: 最多显示点数
        title: 图表标题
        save_path: 保存路径
        dpi: 分辨率
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, rmse_arr in results.items():
        ax.plot(rmse_arr[:max_points], label=label, alpha=0.7)
    ax.set_xlabel('测试样本序号')
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    return fig


# ============================================================
# 逐模态 RMSE 稳定性图
# ============================================================

def plot_per_mode_stability(
    rmse_stream: np.ndarray,
    mode_boundaries: List[int],
    title: str = "逐模态 RMSE 稳定性",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Figure:
    """绘制逐模态 RMSE 稳定性图（带背景色分区）。

    Args:
        rmse_stream: RMSE 序列
        mode_boundaries: 模态边界位置
        title: 图表标题
        save_path: 保存路径
        dpi: 分辨率
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rmse_stream, alpha=0.7, linewidth=0.8, color='black')

    boundaries = [0] + mode_boundaries + [len(rmse_stream)]
    colors = ['#e8f5e9', '#e3f2fd', '#fff3e0', '#fce4ec', '#f3e5f5', '#e0f2f1']
    for i in range(len(boundaries) - 1):
        ax.axvspan(boundaries[i], boundaries[i + 1],
                   alpha=0.3, color=colors[i % len(colors)])
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        seg = rmse_stream[boundaries[i]:boundaries[i + 1]]
        if len(seg) > 0:
            ax.text(mid, ax.get_ylim()[1] * 0.95,
                    f'M{i}\nμ={np.mean(seg):.3f}',
                    ha='center', va='top', fontsize=8)

    for b in mode_boundaries:
        ax.axvline(x=b, color='red', linestyle='--', alpha=0.5)

    ax.set_xlabel('样本序号')
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    fig.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    return fig


# ============================================================
# 变量重要性图
# ============================================================

def plot_variable_importance(
    importance: dict,
    title: str = "变量重要性 (VSN)",
    top_k: int = 15,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Figure:
    """绘制变量选择网络（VSN）的变量重要性柱状图。

    Args:
        importance: compute_variable_importance() 的输出
        title: 图表标题
        top_k: 显示前 K 个最重要的变量
        save_path: 保存路径
        dpi: 分辨率

    Returns:
        matplotlib Figure 对象
    """
    mean_imp = importance['mean']
    std_imp = importance['std']
    names = importance['names']
    sorted_idx = importance['sorted_indices']

    n_show = min(top_k, len(mean_imp))
    show_idx = sorted_idx[:n_show]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(n_show - 1, -1, -1)

    ax.barh(y_pos, mean_imp[show_idx],
            xerr=std_imp[show_idx] if std_imp is not None else None,
            alpha=0.8, color='steelblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in show_idx])
    ax.set_xlabel('平均 VSN 权重')
    ax.set_title(title)
    fig.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    return fig


# ============================================================
# 注意力热力图
# ============================================================

def plot_attention_heatmap(
    attn_weights: np.ndarray,
    title: str = "时序注意力热力图",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Figure:
    """绘制多头注意力的平均热力图。

    Args:
        attn_weights: (num_heads, n_steps, n_steps) 注意力矩阵
        title: 图表标题
        save_path: 保存路径
        dpi: 分辨率

    Returns:
        matplotlib Figure 对象
    """
    avg_attn = attn_weights.mean(axis=0)  # (steps, steps)
    n_steps = avg_attn.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(avg_attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Key 时间步')
    ax.set_ylabel('Query 时间步')
    ax.set_title(title)
    ax.set_xticks(range(0, n_steps, max(1, n_steps // 10)))
    ax.set_yticks(range(0, n_steps, max(1, n_steps // 10)))
    plt.colorbar(im, ax=ax, label='平均注意力权重')
    fig.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    return fig


# ============================================================
# 预测区间图（P10-P50-P90）
# ============================================================

def plot_prediction_interval(
    y_true: np.ndarray,
    y_p10: np.ndarray,
    y_p50: np.ndarray,
    y_p90: np.ndarray,
    feature_idx: int = 0,
    n_points: int = 150,
    title: str = "TFT 预测区间 (P10-P50-P90)",
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Figure:
    """绘制单一特征的 P10-P50-P90 预测区间与真实值对比。

    Args:
        y_true: 真实值 (n_samples, n_features)
        y_p10: P10 预测
        y_p50: P50 预测
        y_p90: P90 预测
        feature_idx: 要绘制的特征索引
        n_points: 最多显示点数
        title: 图表标题
        save_path: 保存路径
        dpi: 分辨率

    Returns:
        matplotlib Figure 对象
    """
    x = np.arange(min(n_points, len(y_true)))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y_true[:n_points, feature_idx], 'k-', linewidth=1.5,
            label='真实值')
    ax.plot(x, y_p50[:n_points, feature_idx], 'b-', linewidth=1.2,
            label='P50 预测')
    ax.fill_between(x,
                    y_p10[:n_points, feature_idx],
                    y_p90[:n_points, feature_idx],
                    alpha=0.2, color='blue', label='P10-P90 区间')
    ax.set_xlabel('样本序号')
    ax.set_ylabel(f'特征 {feature_idx + 1}')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    return fig
