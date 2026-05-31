"""训练与评估 —— 标准训练器、在线检测器、评估与可视化"""

import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple
from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau
from tensorflow.keras.models import load_model

from .models import create_model
from .data import split_sequences, standardize, split_train_test
from .visualization import setup_chinese_font

# 确保中文字体在使用前已配置（模块级自动执行）
setup_chinese_font()


# ============================================================
# 预测辅助：兼容 TFT 分位数输出与标准 MSE 输出
# ============================================================

def _get_p50_prediction(model: tf.keras.Model, y_pred: np.ndarray,
                        config: dict) -> np.ndarray:
    """从模型预测中提取 P50（中位数）用于 RMSE 计算。

    TFT 输出 shape 为 (batch, n_quantiles * n_features)，
    LSTM 输出 shape 为 (batch, n_features)。此函数统一提取为 (batch, n_features)。

    通过检查输出维度自动判定模型类型，无需显式传参。
    """
    n_features = y_pred.shape[1]
    model_type = config.get('MODEL_TYPE', 'lstm')

    if model_type == 'tft':
        n_quantiles = len(config.get('TFT_QUANTILES', [0.1, 0.5, 0.9]))
        expected_per_q = n_features // n_quantiles
        if expected_per_q * n_quantiles == n_features:
            # 确实为分位数输出，提取 P50（索引 1）
            return y_pred[:, expected_per_q:expected_per_q * 2]
    return y_pred


def _auto_n_features(config: dict, input_shape: tuple) -> int:
    """根据模型类型返回输出维度"""
    if config.get('MODEL_TYPE') == 'tft':
        return input_shape[1] * len(config.get('TFT_QUANTILES', [0.1, 0.5, 0.9]))
    return input_shape[1]


# ============================================================
# 评估与可视化
# ============================================================

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方根误差"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_rmse_record(csv_path: str, rmse: float) -> None:
    """追加写入一条 RMSE 记录"""
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".",
                exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([rmse])


def plot_predictions(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    title: str = "预测结果",
    n_points: int = 100,
) -> None:
    """绘制预测值与真实值对比图"""
    x = np.arange(min(n_points, len(y_pred)))
    n_features = y_pred.shape[1]
    for i in range(n_features):
        plt.figure()
        plt.plot(x, y_pred[:n_points, i], linestyle='-', label='预测值')
        plt.plot(x, y_true[:n_points, i], linestyle='-', marker='o', label='真实值')
        plt.xlabel('时间步')
        plt.ylabel(f'特征 {i + 1}')
        plt.title(f'{title} - 特征 {i + 1}')
        plt.legend()
        plt.show()


# ============================================================
# 标准训练器
# ============================================================

def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    verbose: int = 0,
):
    """通用模型训练"""
    return model.fit(X_train, y_train, epochs=epochs,
                     batch_size=batch_size, verbose=verbose)


def run_standard_pipeline(config: dict) -> float:
    """标准 SSA+TFT/LSTM 训练流程"""
    from .data import load_te_mat, decompose_ssa

    data_path = config.get('DEFAULT_DATA_FILE')
    print(f"[数据] 加载: {data_path}")
    df = load_te_mat(data_path)

    # SSA 分解
    print(f"[SSA] 窗口大小 = {config['WINDOW_SIZE']}")
    components = decompose_ssa(df, window_size=config['WINDOW_SIZE'])

    # 划分训练/测试集
    train_raw, test_raw = split_train_test(components.values, ratio=config['TRAIN_RATIO'])
    scaler, train_data, test_data = standardize(train_raw, test_raw)

    # 构造序列
    n_steps = config['N_STEPS']
    X_train, y_train = split_sequences(train_data, n_steps)
    X_test, y_test = split_sequences(test_data, n_steps)

    # 创建模型
    input_shape = (n_steps, train_data.shape[1])
    model_type = config['MODEL_TYPE']
    units = config['TFT_UNITS'] if model_type == 'tft' else config['LSTM_UNITS']
    print(f"[模型] 类型 = {model_type}, 隐藏单元 = {units}")
    model = create_model(model_type, input_shape, units)

    # 训练
    epochs = config['FULL_EPOCHS']
    print(f"[训练] Epochs = {epochs}")
    model.fit(X_train, y_train, epochs=epochs,
              batch_size=config['BATCH_SIZE'], verbose=0)

    # 预测
    y_pred_raw = model.predict(X_test, verbose=0)
    y_pred = _get_p50_prediction(model, y_pred_raw, config)
    y_pred = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)

    # 评估
    rmse = compute_rmse(y_test_inv, y_pred)
    print(f"[评估] RMSE = {rmse:.4f}")

    # 保存最优模型
    if config['SAVE_BEST_MODEL']:
        model_path = os.path.join(config['MODEL_DIR'], 'ssa_tft_best.h5')
        os.makedirs(config['MODEL_DIR'], exist_ok=True)
        rmse_path = os.path.join(config['RESULT_DIR'], 'rmse.csv')
        os.makedirs(config['RESULT_DIR'], exist_ok=True)

        save_new = True
        if os.path.exists(rmse_path):
            with open(rmse_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if float(row[0]) <= rmse:
                        save_new = False
                    break

        if save_new:
            model.save(model_path)
            with open(rmse_path, 'w', newline='') as f:
                csv.writer(f).writerow([rmse])
            print(f"[保存] 模型已保存至 {model_path}")

    plot_predictions(y_pred, y_test_inv, title="SSA-TFT 预测")
    return rmse


# ============================================================
# 辅助：自适应阈值方法
# ============================================================

def _compute_cusum_threshold(
    cal_rmse: List[float],
    drift: float = 0.05,
    reset_after: int = 3,
) -> Tuple[float, callable]:
    """基于校准窗口构建 CUSUM 检测函数。

    Returns:
        (baseline_threshold, cusum_detector_fn)
        cusum_detector_fn(new_rmse) -> (is_alert, current_cusum)
    """
    if len(cal_rmse) < 10:
        mean_r, std_r = np.mean(cal_rmse), np.std(cal_rmse)
        return mean_r + 3 * std_r, None

    baseline = np.mean(cal_rmse)
    std_base = np.std(cal_rmse)
    cusum_threshold = 5 * std_base

    state = {'cusum_pos': 0.0, 'cusum_neg': 0.0}

    def detector(new_rmse: float) -> Tuple[bool, float]:
        deviation = new_rmse - baseline
        state['cusum_pos'] = max(0.0, state['cusum_pos'] + deviation - drift)
        state['cusum_neg'] = max(0.0, state['cusum_neg'] - deviation - drift)
        is_alert = state['cusum_pos'] > cusum_threshold
        return is_alert, state['cusum_pos']

    return baseline + 3 * std_base, detector


def _compute_rolling_percentile_threshold(
    rmse_buffer: List[float],
    window_size: int = 100,
    percentile: float = 99.0,
) -> float:
    """基于滚动窗口的百分位动态阈值"""
    if len(rmse_buffer) < 30:
        return float('inf')
    recent = rmse_buffer[-window_size:]
    return float(np.percentile(recent, percentile))


# ============================================================
# 在线增量检测器
# ============================================================

class OnlineDetector:
    """
    在线模态变化检测器

    算法流程:
    1. 在数据前 init_train_size 个样本上初始训练模型
    2. 动态阈值校准（支持 static / dynamic / cusum / percentile 四种方法）
    3. 逐点预测，计算 RMSE
    4. 若连续 consecutive_count 个点的 RMSE 超过阈值 → 判定模态变化
    5. 模态变化时：回滚到安全检查点（干净模型），从当前位置重新训练
    6. 未超阈值的预测积累到 batch_size 后用于增量微调
    7. 污染检测：RMSE 趋势单调递增时跳过微调
    """

    def __init__(self, config: dict):
        self.n_steps = config['N_STEPS']
        self.threshold = config['THRESHOLD']
        self.consecutive_count = config['CONSECUTIVE_COUNT']
        self.init_train_size = config['INIT_TRAIN_SIZE']
        self.epochs = config['EPOCHS']
        self.batch_size = config['BATCH_SIZE']
        self.model_type = config['MODEL_TYPE']
        self.lstm_units = (config['TFT_UNITS'] if self.model_type == 'tft'
                           else config['LSTM_UNITS'])
        self.model_dir = config['MODEL_DIR']
        self.result_dir = config['RESULT_DIR']

        # 阈值方法
        self.threshold_method = config.get('THRESHOLD_METHOD', 'dynamic')
        self.dynamic_threshold = config.get('DYNAMIC_THRESHOLD', True)
        self.calibration_size = config.get('CALIBRATION_SIZE', 100)
        self.threshold_multiplier = config.get('THRESHOLD_MULTIPLIER', 2.5)
        self.cusum_drift = config.get('CUSUM_DRIFT', 0.05)
        self.rolling_window_size = config.get('ROLLING_WINDOW_SIZE', 100)
        self.threshold_percentile = config.get('THRESHOLD_PERCENTILE', 99)

        # 安全检查点
        self.enable_safe_checkpoint = config.get('ENABLE_SAFE_CHECKPOINT', True)
        self.contamination_window = config.get('CONTAMINATION_WINDOW', 10)
        self.clear_session = config.get('CLEAR_SESSION_ON_TRANSITION', True)

        # 短过渡
        self.enable_bidirectional = config.get('ENABLE_BIDIRECTIONAL_CONFIRM', True)
        self.enable_multiscale = config.get('ENABLE_MULTI_SCALE_SSA', False)
        self.enable_cp_prefilter = config.get('ENABLE_CHANGE_POINT_PREFILTER', False)
        self.min_segment_length = config.get('MIN_SEGMENT_LENGTH', 3)
        self.enable_stat_fallback = config.get('ENABLE_STATISTICAL_FALLBACK', True)

        # 输出
        self.loss_csv = os.path.join(config['RESULT_DIR'], 'loss.csv')
        self.note_csv = os.path.join(config['RESULT_DIR'], 'non_note.csv')
        self.transitions = []

        # 运行时状态
        self._config = config
        self.safe_checkpoint_weights = None
        self.safe_checkpoint_position = 0
        self.rolling_rmse_buffer: List[float] = []
        self.cusum_detector = None
        self.prev_mode_stats: Optional[Dict] = None

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        if os.path.exists(self.loss_csv):
            os.remove(self.loss_csv)

    # ---- 阈值计算 ----

    def _compute_threshold(self, cal_rmse: List[float]) -> float:
        """根据 THRESHOLD_METHOD 计算检测阈值"""
        if len(cal_rmse) == 0:
            return self.threshold

        mean_loss = np.mean(cal_rmse)
        std_loss = np.std(cal_rmse)

        if self.threshold_method == 'static':
            return self.threshold

        elif self.threshold_method == 'cusum':
            baseline, detector = _compute_cusum_threshold(cal_rmse, drift=self.cusum_drift)
            self.cusum_detector = detector
            print(f"[阈值] CUSUM: baseline={baseline:.4f}, "
                  f"cal_mean={mean_loss:.4f}, cal_std={std_loss:.4f}")
            return baseline

        elif self.threshold_method == 'percentile':
            pct = float(np.percentile(cal_rmse, self.threshold_percentile))
            print(f"[阈值] 百分位 ({self.threshold_percentile}%): {pct:.4f}")
            return pct

        else:  # 'dynamic' (default)
            dynamic = mean_loss + self.threshold_multiplier * std_loss
            print(f"[阈值] 动态计算: mean={mean_loss:.4f}, std={std_loss:.4f}, "
                  f"multiplier={self.threshold_multiplier}, threshold={dynamic:.4f}")
            return dynamic

    # ---- 污染检测 ----

    def _detect_contamination(self, rmse_values: List[float]) -> bool:
        """检测 RMSE 趋势是否单调递增（模型正在被污染）。"""
        if len(rmse_values) < self.contamination_window:
            return False
        recent = rmse_values[-self.contamination_window:]
        tau, p = kendalltau(range(len(recent)), recent)
        return tau > 0.5 and p < 0.05

    # ---- 安全检查点 ----

    def _save_safe_checkpoint(self, model: tf.keras.Model, position: int) -> None:
        """保存干净模型的权重到内存"""
        if self.enable_safe_checkpoint:
            self.safe_checkpoint_weights = model.get_weights()
            self.safe_checkpoint_position = position

    def _restore_safe_checkpoint(self, model: tf.keras.Model) -> int:
        """恢复模型到上一个安全检查点"""
        if self.safe_checkpoint_weights is not None:
            model.set_weights(self.safe_checkpoint_weights)
            return self.safe_checkpoint_position
        return -1

    # ---- 模态变化确认 ----

    def _predict_and_rmse(self, model: tf.keras.Model, X: np.ndarray,
                          y_true: np.ndarray) -> float:
        """统一预测并计算 RMSE（自动处理 TFT 分位数输出）"""
        y_pred_raw = model.predict(X, verbose=0)
        y_pred = _get_p50_prediction(model, y_pred_raw[0:1], self._config)
        return compute_rmse(y_true[0], y_pred[0])

    def _confirm_mode_change(
        self,
        model: tf.keras.Model,
        X_all: np.ndarray,
        y_all: np.ndarray,
        trigger_pos: int,
    ) -> bool:
        """检查触发点之后 consecutive_count-1 个点是否也超过阈值。

        若启用双向确认，还会从触发点之后训练反向模型交叉验证。
        """
        check_end = min(trigger_pos + self.consecutive_count, len(X_all))
        for i in range(trigger_pos + 1, check_end):
            X = X_all[i:i + 1]
            y_true = y_all[i:i + 1]
            rmse = self._predict_and_rmse(model, X, y_true)
            save_rmse_record(self.loss_csv, rmse)

            # CUSUM 模式下使用累积和判定
            if self.cusum_detector is not None:
                is_alert, _ = self.cusum_detector(rmse)
                if not is_alert:
                    return False
            elif rmse <= self.threshold:
                return False

        # 双向确认
        if self.enable_bidirectional and trigger_pos > self.n_steps + 10:
            try:
                from .short_transition import bidirectional_confirm
                return bidirectional_confirm(
                    model, X_all, y_all,
                    trigger_pos=trigger_pos,
                    threshold=self.threshold,
                    n_steps=self.n_steps,
                    batch_size=self.batch_size,
                )
            except Exception:
                pass

        return True

    # ---- 短片段处理 ----

    def _handle_short_segment(
        self,
        segment_data: np.ndarray,
        prev_mode_mean: np.ndarray,
        prev_mode_cov: np.ndarray,
    ) -> dict:
        """对极短片段使用马氏距离做统计分类"""
        from .short_transition import classify_short_segment
        return classify_short_segment(
            segment_data, prev_mode_mean, prev_mode_cov,
            threshold=3.0,
        )

    # ---- 主检测循环 ----

    def run(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        gpu_device: Optional[str] = None,
    ) -> List[int]:
        """运行在线检测。

        Args:
            X_all: (n_samples, n_steps, n_features) 输入序列
            y_all: (n_samples, n_features) 标签
            gpu_device: GPU 设备字符串或 None

        Returns:
            检测到的模态变化位置列表
        """
        total = len(X_all)
        pos = 0
        n_features = X_all.shape[2]

        while pos + self.init_train_size < total:
            segment_start = pos
            train_end = pos + self.init_train_size

            # ---- 初始训练 ----
            X_init = X_all[segment_start:train_end]
            y_init = y_all[segment_start:train_end]
            input_shape = (self.n_steps, n_features)
            model = create_model(self.model_type, input_shape, self.lstm_units)

            try:
                if gpu_device:
                    with tf.device(gpu_device):
                        model.fit(X_init, y_init, epochs=self.epochs,
                                  batch_size=self.batch_size, verbose=0)
                else:
                    model.fit(X_init, y_init, epochs=self.epochs,
                              batch_size=self.batch_size, verbose=0)
            except Exception as e:
                print(f"[错误] 初始训练失败 @ 位置 {segment_start}: {e}")
                pos += 1
                continue

            # 保存安全检查点
            self._save_safe_checkpoint(model, train_end)
            # 记录前一模态统计信息（用于短片段回退）
            self.prev_mode_stats = {
                'mean': np.mean(y_init, axis=0),
                'cov': np.cov(y_init, rowvar=False),
            }

            # ---- 阈值校准 ----
            cursor = train_end
            buffer_X, buffer_y = [], []
            self.rolling_rmse_buffer = []

            if self.calibration_size > 0:
                cal_end = min(train_end + self.calibration_size, total)
                cal_rmse = []
                for i in range(train_end, cal_end):
                    try:
                        X_cur = X_all[i:i + 1]
                        y_cur = y_all[i:i + 1]
                        rmse = self._predict_and_rmse(model, X_cur, y_cur)
                    except Exception:
                        rmse = float('inf')
                    cal_rmse.append(rmse)
                    save_rmse_record(self.loss_csv, rmse)
                    self.rolling_rmse_buffer.append(rmse)

                    buffer_X.append(X_cur[0])
                    buffer_y.append(y_cur[0])
                    if len(buffer_y) >= self.batch_size:
                        bx = np.array(buffer_X)
                        byb = np.array(buffer_y)
                        try:
                            if gpu_device:
                                with tf.device(gpu_device):
                                    model.fit(bx, byb, epochs=self.epochs,
                                              batch_size=self.batch_size, verbose=0)
                            else:
                                model.fit(bx, byb, epochs=self.epochs,
                                          batch_size=self.batch_size, verbose=0)
                        except Exception:
                            pass
                        buffer_X, buffer_y = [], []

                self.threshold = self._compute_threshold(cal_rmse)
                cursor = cal_end

            # ---- 逐点预测 ----
            transition_detected = False
            local_rmse_history: List[float] = []

            while cursor < total:
                try:
                    X_cur = X_all[cursor:cursor + 1]
                    y_cur = y_all[cursor:cursor + 1]
                    rmse = self._predict_and_rmse(model, X_cur, y_cur)
                except Exception:
                    cursor += 1
                    continue

                save_rmse_record(self.loss_csv, rmse)
                self.rolling_rmse_buffer.append(rmse)

                # 百分位模式：动态更新阈值
                if self.threshold_method == 'percentile':
                    self.threshold = _compute_rolling_percentile_threshold(
                        self.rolling_rmse_buffer,
                        window_size=self.rolling_window_size,
                        percentile=self.threshold_percentile,
                    )

                # CUSUM 检测
                cusum_alert = False
                if self.cusum_detector is not None:
                    cusum_alert, _ = self.cusum_detector(rmse)

                is_anomaly = (rmse > self.threshold) or cusum_alert

                if is_anomaly:
                    local_rmse_history.append(rmse)
                    if self._confirm_mode_change(model, X_all, y_all, cursor):
                        # 模态变化确认
                        if self.enable_safe_checkpoint:
                            restored_pos = self._restore_safe_checkpoint(model)
                            if restored_pos >= 0:
                                model_path = os.path.join(
                                    self.model_dir, f'lstm_model_{restored_pos}.h5')
                            else:
                                model_path = os.path.join(
                                    self.model_dir, f'lstm_model_{cursor}.h5')
                        else:
                            model_path = os.path.join(
                                self.model_dir, f'lstm_model_{cursor}.h5')

                        model.save(model_path)
                        self.transitions.append(cursor)
                        with open(self.note_csv, 'a', newline='') as f:
                            csv.writer(f).writerow([cursor, rmse])
                        print(f"[检测] 模态变化 @ 样本 {cursor}, RMSE = {rmse:.4f}")
                        pos = cursor
                        transition_detected = True

                        if self.clear_session:
                            tf.keras.backend.clear_session()
                        break
                    else:
                        local_rmse_history = []
                        cursor += 1
                        continue
                else:
                    local_rmse_history = []

                # 污染检测：跳过受污染样本的微调
                if (self.enable_safe_checkpoint
                        and self._detect_contamination(local_rmse_history)):
                    cursor += 1
                    continue

                # 积累正常预测进行增量微调
                buffer_X.append(X_cur[0])
                buffer_y.append(y_cur[0])
                if len(buffer_y) >= self.batch_size:
                    bx = np.array(buffer_X)
                    byb = np.array(buffer_y)
                    try:
                        if gpu_device:
                            with tf.device(gpu_device):
                                model.fit(bx, byb, epochs=self.epochs,
                                          batch_size=self.batch_size, verbose=0)
                        else:
                            model.fit(bx, byb, epochs=self.epochs,
                                      batch_size=self.batch_size, verbose=0)
                    except Exception:
                        pass
                    buffer_X, buffer_y = [], []

                    # 更新安全检查点
                    self._save_safe_checkpoint(model, cursor)

                cursor += 1

            if not transition_detected:
                model_path = os.path.join(self.model_dir, f'lstm_model_final.h5')
                model.save(model_path)
                print(f"[完成] 检测结束，共发现 {len(self.transitions)} 次模态变化")
                print(f"  变化位置: {self.transitions}")
                break

        return self.transitions
