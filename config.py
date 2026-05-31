"""SSA-TFT 项目配置文件 —— 所有超参数和路径的集中管理"""

import argparse
import os


# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataSet", "TE transition mode data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

# 默认数据文件
DEFAULT_DATA_FILE = os.path.join(DATA_DIR, "M1M2XMEAS.mat")

# ============================================================
# SSA 参数
# ============================================================
WINDOW_SIZE = 12          # SSA 滞后窗口大小

# ============================================================
# 序列构造参数
# ============================================================
N_STEPS = 12              # LSTM 输入序列长度（用前 N 步预测下一步）

# ============================================================
# 模型参数
# ============================================================
LSTM_UNITS = 50           # LSTM 隐藏层单元数（对比基线用）
MODEL_TYPE = "tft"        # 默认模型: "tft"（核心方法），可选 "lstm", "attention_lstm", "robust_lstm", "control_gate_lstm"

# ============================================================
# TFT 参数
# ============================================================
TFT_UNITS = 32            # TFT LSTM 编码器隐藏单元数
TFT_HIDDEN_DIM = 64       # GRN / 全连接层隐藏维度
TFT_NUM_HEADS = 4         # 多头注意力头数
TFT_DROPOUT = 0.1         # Dropout 比例
TFT_LSTM_LAYERS = 1       # TFT 内部 LSTM 层数
TFT_QUANTILES = [0.1, 0.5, 0.9]  # 分位数输出列表
TFT_STATIC_FEATURES = 0   # 静态特征数量（TE 数据默认为 0）

# ============================================================
# 训练参数
# ============================================================
EPOCHS = 50               # 增量训练轮数（在线模式）
FULL_EPOCHS = 10000       # 标准训练轮数（train 模式）
BATCH_SIZE = 32           # 批次大小
TRAIN_RATIO = 0.8         # 训练/测试分割比例
SEED = 11                 # 随机种子

# ============================================================
# 在线检测参数
# ============================================================
INIT_TRAIN_SIZE = 200     # 初始训练窗口大小
THRESHOLD = 1.5           # RMSE 阈值（静态模式；DYNAMIC_THRESHOLD=True 时自动覆盖）
CONSECUTIVE_COUNT = 3     # 连续超阈值次数（达到此次数判定模态确实变化）
DYNAMIC_THRESHOLD = True  # 是否使用动态阈值（基于校准窗口 RMSE 均值和标准差）
CALIBRATION_SIZE = 100    # 动态阈值校准窗口大小（训练后预测N个未见样本来估算正常RMSE）
THRESHOLD_MULTIPLIER = 2.5  # 动态阈值乘数：threshold = mean(RMSE) + multiplier * std(RMSE)

# ============================================================
# 短过渡检测参数
# ============================================================
ENABLE_MULTI_SCALE_SSA = False      # 是否启用多尺度 SSA 分解
SSA_WINDOW_SIZES = [6, 12, 24]      # 多尺度 SSA 的窗口大小列表
ENABLE_BIDIRECTIONAL_CONFIRM = True # 是否启用双向残差确认
ENABLE_CHANGE_POINT_PREFILTER = False  # 是否启用变化点预筛选（需 ruptures 库）

# ============================================================
# 自适应阈值参数
# ============================================================
THRESHOLD_METHOD = "dynamic"   # 阈值方法: "static", "dynamic", "cusum", "percentile"
CUSUM_DRIFT = 0.05             # CUSUM 漂移参数
ROLLING_WINDOW_SIZE = 100      # 滚动百分位窗口大小
THRESHOLD_PERCENTILE = 99      # 滚动百分位阈值

# ============================================================
# 安全检查点与污染检测
# ============================================================
ENABLE_SAFE_CHECKPOINT = True     # 模态变化时回滚到过渡前的干净模型
CONTAMINATION_WINDOW = 10         # 污染趋势检测窗口大小
CLEAR_SESSION_ON_TRANSITION = True  # 模态变化时清理 TF 计算图

# ============================================================
# 短片段处理
# ============================================================
MIN_SEGMENT_LENGTH = 3             # LSTM 分类所需的最小样本数
ENABLE_STATISTICAL_FALLBACK = True # 极短片段使用马氏距离统计回退

# ============================================================
# 输出
# ============================================================
SAVE_BEST_MODEL = True    # 是否保存最优模型


def validate_config(config: dict) -> dict:
    """校验配置参数，对不合理的组合抛出明确的错误"""
    if config.get('INIT_TRAIN_SIZE', 0) < config.get('N_STEPS', 0):
        raise ValueError(
            f"INIT_TRAIN_SIZE ({config.get('INIT_TRAIN_SIZE')}) 必须 >= "
            f"N_STEPS ({config.get('N_STEPS')})，否则无法构造训练序列。"
        )
    if config.get('CALIBRATION_SIZE', 0) < config.get('CONSECUTIVE_COUNT', 0):
        raise ValueError(
            f"CALIBRATION_SIZE ({config.get('CALIBRATION_SIZE')}) 必须 >= "
            f"CONSECUTIVE_COUNT ({config.get('CONSECUTIVE_COUNT')})。"
        )
    if config.get('WINDOW_SIZE', 0) <= 1:
        raise ValueError("WINDOW_SIZE 必须 > 1。")
    if config.get('BATCH_SIZE', 0) <= 0:
        raise ValueError("BATCH_SIZE 必须 > 0。")
    if config.get('TRAIN_RATIO', 0) <= 0 or config.get('TRAIN_RATIO', 0) >= 1:
        raise ValueError("TRAIN_RATIO 必须在 (0, 1) 之间。")
    return config


def load_config_from_yaml(yaml_path: str) -> dict:
    """从 YAML 文件加载配置，覆盖默认值"""
    import yaml

    with open(yaml_path, 'r') as f:
        overrides = yaml.safe_load(f) or {}

    base = {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith("_")}
    base.update(overrides)
    return validate_config(base)


def get_config(args=None) -> dict:
    """根据命令行参数更新配置，返回配置字典"""
    config = {k: v for k, v in globals().items()
              if k.isupper() and not k.startswith("_")}

    if args is None:
        return config

    for key, value in vars(args).items():
        upper_key = key.upper()
        if value is not None and upper_key in config:
            config[upper_key] = value

    return config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SSA-TFT TE过程模态变化检测")

    parser.add_argument("--mode", choices=["train", "online", "compare"],
                        default="online", help="运行模式")
    parser.add_argument("--data", default=None,
                        help="数据文件路径（覆盖默认值）")
    parser.add_argument("--model", choices=["lstm", "attention_lstm", "robust_lstm", "control_gate_lstm", "tft"],
                        default=None, help="模型类型")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数")
    parser.add_argument("--threshold", type=float, default=None,
                        help="在线检测RMSE阈值")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--config", default=None,
                        help="YAML 配置文件路径（加载后命令行参数仍可覆盖）")

    return parser.parse_args()
