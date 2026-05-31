# SSA-TFT: TE过程模态变化在线检测

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于**奇异谱分析（SSA）+ Temporal Fusion Transformer（TFT）**的田纳西-伊斯曼（Tennessee Eastman, TE）化工过程模态变化在线检测方法。

> 本科生毕业论文课题 | 2026年5月

## 目录

- [研究背景](#研究背景)
- [技术路线](#技术路线)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [运行模式](#运行模式)
- [配置说明](#配置说明)
- [模型介绍](#模型介绍)
- [自适应阈值策略](#自适应阈值策略)
- [短过渡检测](#短过渡检测)
- [评估体系](#评估体系)
- [数据集](#数据集)
- [实验配置](#实验配置)
- [依赖项](#依赖项)
- [引用](#引用)

---

## 研究背景

田纳西-伊斯曼（Tennessee Eastman, TE）过程是化工过程控制领域的标准基准测试平台。实际化工生产过程中，工况模态（操作模式）的切换会导致过程变量统计特性发生显著变化。及时、准确地检测模态变化，对过程安全监控和故障预警具有重要意义。

本项目提出了一种**SSA降噪 + TFT时序预测 + 在线RMSE监控**的模态变化检测框架，能够在TE过程数据流中实时检测操作模态的切换。

### 主要创新点

- **SSA预处理**：利用奇异谱分析对TE过程数据进行降噪和主成分提取，相比直接使用原始变量，提高了预测模型对模态变化的敏感度
- **在线增量学习**：检测器在正常运行期间持续微调模型，模态变化时自动重新训练，适应过程动态特性
- **多模型支持**：实现了5种时序预测模型（LSTM / Attention-LSTM / Robust-LSTM / Control-Gate-LSTM / TFT），可灵活对比选择
- **多策略自适应阈值**：支持静态、动态统计、CUSUM累积和、滚动百分位四种阈值方法

---

## 技术路线

```
TE多变量时序数据
       │
       ▼
SSA奇异谱分析（降噪+特征提取）
       │
       ▼
滑动窗口构造序列
       │
       ▼
TFT预测模型（分位数输出）
       │
       ▼
逐点预测 + RMSE监控
       │
       ▼
连续超阈值判定模态变化
       │
       ▼
增量微调 + 重新训练 → 继续在线检测
```

**核心流程**：

1. **SSA分解**：使用截断SVD对多变量时序数据进行奇异谱分解，提取主成分（降噪同时保留趋势特征）
2. **序列构造**：滑动窗口将时序转换为监督学习格式（前N步→预测下一步）
3. **模型预测**：TFT（或LSTM变体）对下一步进行预测，输出分位数
4. **RMSE监控**：逐点计算预测值与真实值的均方根误差
5. **模态变化判定**：RMSE连续超阈值达到指定次数时，判定模态发生改变
6. **自适应**：检测到模态变化后自动重新训练，适应新工况

---

## 项目结构

```
SSA-TFT/
├── main.py                         # CLI统一入口（3种运行模式）
├── config.py                       # 集中配置管理（50+参数）
├── requirements.txt                # Python依赖
│
├── src/                            # 核心源码
│   ├── data.py                     # 数据加载、SSA分解、序列构造
│   ├── models.py                   # 5种模型实现 + 模型工厂
│   ├── trainer.py                  # 标准训练器 + 在线检测器
│   ├── short_transition.py         # 短过渡检测专项模块
│   ├── evaluation.py               # 评估指标与报告
│   ├── visualization.py            # 统一可视化
│   ├── experiment_tracker.py       # 实验追踪
│   ├── logger.py                   # 结构化日志
│   └── utils.py                    # 随机种子、GPU检测
│
├── experiments/                    # 实验脚本
│   ├── run_train.py                # 标准训练
│   ├── run_online.py               # 在线模态检测（核心）
│   ├── run_compare.py              # 多模型对比
│   └── run_batch.py                # 批量实验运行器
│
├── configs/                        # 实验配置文件
│   ├── baseline.yaml               # 基线实验配置
│   ├── short_transition.yaml       # 短过渡检测配置
│   ├── tft_baseline.yaml           # TFT基线配置
│   ├── tft_full.yaml               # TFT完整配置
│   ├── ablation.yaml               # 消融实验配置
│   └── ground_truth.yaml           # 地面真值标注
│
├── dataSet/                        # TE过程数据集
│   └── TE transition mode data/    # 模态转换数据（30+ .mat文件）
│
├── outputs/                        # 输出结果
│   ├── models/                     # 已训练模型 (*.h5)
│   ├── results/                    # RMSE记录、评估报告
│   └── experiments/                # 实验追踪目录
│
└── legacy/                         # 历史实验代码
```

---

## 快速开始

### 环境要求

- Python 3.9+
- TensorFlow 2.10+
- Windows / Linux / macOS

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/SSA-TFT.git
cd SSA-TFT

# 2. 创建虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 快速验证

```bash
# 使用基线配置运行在线检测（快速验证，epochs较少）
python main.py --config configs/baseline.yaml --mode online --data M1M2XMEAS.mat
```

---

## 运行模式

项目支持三种运行模式，通过 `--mode` 参数切换：

### 1. train — 标准训练模式

使用SSA预处理后的数据训练模型，自动保存最优模型。

```bash
# 使用默认配置训练
python main.py --mode train

# 指定模型类型和训练轮数
python main.py --mode train --model tft --epochs 5000

# 使用YAML配置文件
python main.py --mode train --config configs/tft_full.yaml
```

### 2. online — 在线模态变化检测（核心模式）

模拟在线数据流，逐点预测并监控RMSE，当RMSE连续异常时判定模态发生变化。

```bash
# 基线实验：LSTM + 动态阈值
python main.py --config configs/baseline.yaml --mode online --data M1M2XMEAS.mat

# TFT模型 + 完整配置
python main.py --config configs/tft_full.yaml --mode online

# 启用短过渡检测增强
python main.py --config configs/short_transition.yaml --mode online

# 自定义阈值
python main.py --mode online --threshold 2.0 --model tft
```

### 3. compare — 模型对比模式

在相同数据集上对比多种模型的预测性能。

```bash
# 对比所有模型
python main.py --mode compare --epochs 50

# 指定数据集
python main.py --mode compare --data M2M4XMEAS.mat
```

---

## 配置说明

所有参数集中管理在 `config.py` 中，可通过 YAML 配置文件和命令行参数覆盖。

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_TYPE` | `"tft"` | 模型类型：`lstm`, `attention_lstm`, `robust_lstm`, `control_gate_lstm`, `tft` |
| `WINDOW_SIZE` | `12` | SSA奇异谱分析滞后窗口大小 |
| `N_STEPS` | `12` | LSTM输入序列长度（前N步预测下一步） |
| `LSTM_UNITS` | `50` | LSTM隐藏层单元数 |
| `EPOCHS` | `50` | 在线模式增量训练轮数 |
| `FULL_EPOCHS` | `10000` | 标准训练轮数 |
| `BATCH_SIZE` | `32` | 批次大小 |
| `TRAIN_RATIO` | `0.8` | 训练/测试分割比例 |
| `SEED` | `11` | 随机种子 |

### TFT参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TFT_UNITS` | `32` | LSTM编码器隐藏单元数 |
| `TFT_HIDDEN_DIM` | `64` | GRN/全连接层隐藏维度 |
| `TFT_NUM_HEADS` | `4` | 多头注意力头数 |
| `TFT_DROPOUT` | `0.1` | Dropout比例 |
| `TFT_LSTM_LAYERS` | `1` | TFT内部LSTM层数 |
| `TFT_QUANTILES` | `[0.1, 0.5, 0.9]` | 分位数输出列表 |

### 在线检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `INIT_TRAIN_SIZE` | `200` | 初始训练窗口大小 |
| `THRESHOLD` | `1.5` | 静态RMSE阈值 |
| `CONSECUTIVE_COUNT` | `3` | 连续超阈值次数（判定模态变化） |
| `DYNAMIC_THRESHOLD` | `True` | 是否启用动态阈值 |
| `CALIBRATION_SIZE` | `100` | 动态阈值校准窗口大小 |
| `THRESHOLD_MULTIPLIER` | `2.5` | 动态阈值乘数 |
| `THRESHOLD_METHOD` | `"dynamic"` | 阈值方法 |

### YAML配置文件

可以使用 YAML 文件覆写任何配置参数：

```yaml
# configs/baseline.yaml
MODEL_TYPE: "lstm"
LSTM_UNITS: 50
WINDOW_SIZE: 12
N_STEPS: 12
EPOCHS: 50
FULL_EPOCHS: 10000
BATCH_SIZE: 32
INIT_TRAIN_SIZE: 200
THRESHOLD_METHOD: "dynamic"
DYNAMIC_THRESHOLD: true
CALIBRATION_SIZE: 100
THRESHOLD_MULTIPLIER: 2.5
CONSECUTIVE_COUNT: 3
SEED: 11
```

命令行参数优先级高于YAML配置，可以实现"YAML定义基座 + 命令行覆盖"的灵活组合。

---

## 模型介绍

### 1. 标准LSTM（Baseline）

单层LSTM + Dense输出层，作为对比基线。

### 2. Attention-LSTM

内嵌**Bahdanau注意力机制**的自定义LSTM单元。在隐藏状态更新时，对历史时间步计算注意力权重，重点关注与当前预测相关的时间步。

### 3. Robust-LSTM

**自适应遗忘门**：将当前输入与期望值的差异融入遗忘门计算，使模型对异常值具有更好的鲁棒性，在模态过渡期间表现更稳定。

### 4. Control-Gate-LSTM

**双重门控机制**：遗忘门 + 控制门协同调节细胞状态更新，提升对复杂时序依赖关系的建模能力。

### 5. TFT (Temporal Fusion Transformer)

完整的时序Transformer架构，包含：

- **Variable Selection Network**：自适应选择最相关的输入变量
- **LSTM Encoder**：捕获局部时序模式
- **Multi-Head Self-Attention**：建模长程依赖关系
- **Gated Residual Network (GRN)**：非线性特征变换 + 门控跳过连接
- **Quantile Output**：输出分位数预测（默认 [0.1, 0.5, 0.9]），提供预测不确定性估计

```python
from src.models import create_model

# 通过模型工厂统一创建
model = create_model(
    model_type='tft',
    input_shape=(12, 52),  # (N_STEPS, n_features)
    config=config
)
```

---

## 自适应阈值策略

| 方法 | 配置值 | 原理 | 适用场景 |
|------|--------|------|----------|
| **Static** | `"static"` | 固定阈值，由 `THRESHOLD` 参数指定 | 已知正常RMSE范围 |
| **Dynamic** | `"dynamic"` | `μ(RMSE) + k·σ(RMSE)`，在校准窗口上估计正常RMSE分布 | 一般情况（**默认推荐**） |
| **CUSUM** | `"cusum"` | 累积和检测小幅持续漂移，对渐变变化敏感 | 渐变模态变化 |
| **Percentile** | `"percentile"` | 滚动窗口P99动态阈值，自动适应噪声水平变化 | 噪声水平变化大的场景 |

### 动态阈值原理

```
threshold = mean(RMSE_calibration) + THRESHOLD_MULTIPLIER × std(RMSE_calibration)
```

在训练完成后的 `CALIBRATION_SIZE` 个未见样本上计算RMSE的均值和标准差，以此设定自适应的异常阈值。

---

## 短过渡检测

针对过渡时间小于SSA窗口大小的挑战，系统提供三个增强模块：

### 多尺度SSA

同时运行多个窗口大小的SSA分解（如 W=6, 12, 24），短过渡在细粒度窗口（W=6）中表现为尖锐峰值，更容易被检测。

### 双向残差确认

模态边界从前后两个方向进行交叉验证——真正的模态变化两侧（边界前和边界后）都应对应RMSE异常，从而过滤单侧噪声。

### 变化点预筛选

基于 [ruptures](https://github.com/deepcharles/ruptures) 库的 PELT 算法对信号进行粗筛，筛出的候选变化点再由LSTM模型进行精细确认。

```yaml
# configs/short_transition.yaml
ENABLE_MULTI_SCALE_SSA: true
SSA_WINDOW_SIZES: [6, 12, 24]
ENABLE_BIDIRECTIONAL_CONFIRM: true
ENABLE_CHANGE_POINT_PREFILTER: false  # 需安装 ruptures 库
```

### 安全检查点机制

- 正常段定期保存模型权重到内存
- 模态变化确认时回滚到过渡**前**的干净检查点，防止过渡数据污染模型
- Kendall tau污染趋势检测：RMSE单调递增时跳过增量微调
- 模态变化后清理TF计算图，释放显存

---

## 评估体系

系统提供完整的评估指标和可视化：

### 检测精度指标

- **Precision / Recall / F1**：基于地面真值的模态变化点检测精度（带容差窗口匹配）
- **检测延迟**：真实过渡到首次检测的样本数（均值 / 中位数 / 最大值）
- **阈值敏感性AUC**：扫描 `THRESHOLD_MULTIPLIER`，绘制 TPR vs FPR 曲线

### 模型质量指标

- **逐模态RMSE稳定性**：每个模态段内的RMSE均值/标准差
- **污染分数**：增量微调模型 vs 从头重训模型的退化比例

### 可视化

- RMSE时序流图（含阈值线和检测点标注）
- 检测延迟分布直方图
- 阈值敏感性ROC曲线
- 多模型预测RMSE对比图
- 模态分区预测值vs真实值对比图

---

## 数据集

TE过程模态转换数据存放在 `dataSet/TE transition mode data/` 目录下。

### 数据文件

| 模态转换对 | 测量变量文件 | 操纵变量文件 |
|-----------|-------------|-------------|
| Mode 1 ↔ Mode 2 | `M1M2XMEAS.mat`, `M2M1XMEAS.mat` | `M1M2XMV.mat`, `M2M1XMV.mat` |
| Mode 1 ↔ Mode 4 | `M1M4XMEAS.mat`, `M4M1XMEAS.mat` | `M1M4XMV.mat`, `M4M1XMV.mat` |
| Mode 1 ↔ Mode 5 | `M1M5XMEAS.mat`, `M5M1XMEAS.mat` | `M1M5XMV.mat`, `M5M1XMV.mat` |
| Mode 2 ↔ Mode 4 | `M2M4XMEAS.mat`, `M4M2XMEAS.mat` | `M2M4XMV.mat`, `M4M2XMV.mat` |
| Mode 2 ↔ Mode 5 | `M2M5XMEAS.mat`, `M5M2XMEAS.mat` | `M2M5XMV.mat`, `M5M2XMV.mat` |
| Mode 3 ↔ Mode 6 | `M3M6XMEAS.mat`, `M6M3XMEAS.mat` | `M3M6XMV.mat`, `M6M3XMV.mat` |
| 异常工况 | `M4M2XMEAS(ABNORMAL).mat` | `M4M2XMV(ABNORMAL).mat` |

- **XMEAS**：41个测量变量（如温度、压力、液位、流量等）
- **XMV**：11个操纵变量（阀门开度、设定值等）

### 地面真值

`configs/ground_truth.yaml` 记录了各数据文件中的真实模态转换点位置，用于评估检测精度。

---

## 实验配置

### 基线实验

```bash
python main.py --config configs/baseline.yaml --mode online --data M1M2XMEAS.mat
```

使用标准LSTM + 动态阈值，验证基础检测流程。

### TFT完整实验

```bash
python main.py --config configs/tft_full.yaml --mode online
```

启用TFT模型 + 安全检查点 + 所有在线检测增强功能。

### 短过渡检测实验

```bash
python main.py --config configs/short_transition.yaml --mode online
```

启用多尺度SSA + 双向残差确认，验证短过渡检测能力。

### 消融实验

```bash
python main.py --config configs/ablation.yaml --mode online
```

逐步关闭各增强组件，分析每个模块的贡献。

### 批量实验

```python
# 阈值敏感性扫描
from experiments.run_batch import run_threshold_sweep
from config import get_config

cfg = get_config()
results = run_threshold_sweep(cfg)

# 全模型对比
from experiments.run_batch import run_model_comparison_full
results = run_model_comparison_full(cfg)
```

---

## 依赖项

```
tensorflow>=2.10.0    # 深度学习框架
numpy>=1.21.0         # 数值计算
pandas>=1.3.0         # 数据处理
scipy>=1.7.0          # 科学计算（Kendall tau等）
scikit-learn>=1.0.0   # 机器学习工具（标准化等）
matplotlib>=3.5.0     # 可视化
pyyaml>=6.0           # YAML配置文件解析
ruptures>=0.1.7       # 变化点检测（PELT算法，可选）
```

---

## 项目统计

| 指标 | 数值 |
|------|------|
| 核心源码文件 | 9个 (`src/`) |
| 实验脚本 | 4个 (`experiments/`) |
| 模型类型 | 5种 |
| 阈值方法 | 4种 |
| 配置参数 | 50+ |
| 数据集文件 | 30+ .mat |
| 代码总行数 | ~2500行 |
| 支持运行模式 | 3种 (train / online / compare) |

---

## 引用

如果本项目对你的研究有帮助，欢迎引用：

```bibtex
@software{ssa-tft-2026,
  title     = {{SSA-TFT}: Online Mode Change Detection for Tennessee Eastman Process
               using Singular Spectrum Analysis and Temporal Fusion Transformer},
  year      = {2026},
  url       = {https://github.com/your-username/SSA-TFT}
}
```

---

## 许可证

本项目采用 [MIT License](LICENSE)。学术用途欢迎自由使用和修改，商业用途请联系作者。

---

## 作者

本科生毕业论文课题 | 武汉纺织大学计算机与人工智能学院

如有问题或建议，欢迎提交 [Issue](../../issues)。
