"""SSA-TFT: Temporal Fusion Transformer 模块

完整的 TFT 实现，包含：
- Gated Residual Network (GRN) 与 Gated Linear Unit (GLU)
- Variable Selection Network (VSN) —— 逐时间步特征选择
- Static Covariate Encoder —— 时不变特征编码
- Causal Temporal Self-Attention —— 因果时序自注意力
- 分位数输出 (P10/P50/P90)
- 可解释性分析（变量重要性 + 注意力权重）
"""

from .components import build_tft, quantile_loss_fn, \
    extract_p50_prediction, extract_prediction_intervals
from .gated import GatedResidualNetwork, GLU
from .vsn import VariableSelectionNetwork, TimeDistributedVSN
from .static_encoder import StaticCovariateEncoder
from .attention import CausalTemporalSelfAttention
from .interpret import (
    get_variable_selection_weights,
    get_attention_weights,
    compute_variable_importance,
    compute_attention_importance,
)

__all__ = [
    'build_tft',
    'quantile_loss_fn',
    'extract_p50_prediction',
    'extract_prediction_intervals',
    'GatedResidualNetwork',
    'GLU',
    'VariableSelectionNetwork',
    'TimeDistributedVSN',
    'StaticCovariateEncoder',
    'CausalTemporalSelfAttention',
    'get_variable_selection_weights',
    'get_attention_weights',
    'compute_variable_importance',
    'compute_attention_importance',
]
