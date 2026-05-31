"""
SSA-TFT TE过程模态变化检测 —— 统一入口

用法:
    python main.py --mode train            # 标准 SSA+TFT 训练
    python main.py --mode online           # 在线模态变化检测
    python main.py --mode compare          # 模型对比（TFT vs LSTM变体）
    python main.py --mode train --model attention_lstm --data M1M2XMEAS.mat
    python main.py --config configs/tft_baseline.yaml  # 使用 YAML 配置文件
"""

import sys
import os

from config import (parse_args, get_config, load_config_from_yaml,
                    validate_config, DEFAULT_DATA_FILE)
from src.utils import set_global_determinism


def main():
    args = parse_args()

    # 加载 YAML 配置（如有指定）
    if args.config:
        if not os.path.exists(args.config):
            print(f"配置文件不存在: {args.config}")
            sys.exit(1)
        config = load_config_from_yaml(args.config)
        # 命令行参数覆盖 YAML 配置
        for key, value in vars(args).items():
            upper_key = key.upper()
            if value is not None and upper_key in config:
                config[upper_key] = value
    else:
        config = get_config(args)
        config = validate_config(config)

    # 确保使用绝对路径
    if config.get('DEFAULT_DATA_FILE') is None:
        config['DEFAULT_DATA_FILE'] = DEFAULT_DATA_FILE

    # 如果用户通过 --data 指定了文件名而不是完整路径，补全路径
    data_arg = config.get('DEFAULT_DATA_FILE')
    if data_arg and not os.path.isabs(data_arg):
        config['DEFAULT_DATA_FILE'] = os.path.join(
            os.path.dirname(DEFAULT_DATA_FILE), data_arg
        )

    set_global_determinism(seed=config['SEED'])

    mode = args.mode

    if mode == 'train':
        from experiments.run_train import run
    elif mode == 'online':
        from experiments.run_online import run
    elif mode == 'compare':
        from experiments.run_compare import run
    else:
        print(f"未知模式: {mode}")
        sys.exit(1)

    run(config)


if __name__ == "__main__":
    main()
