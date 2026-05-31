"""实验追踪模块 —— 运行记录、参数快照、指标日志、制品管理"""

import os
import json
import csv
from datetime import datetime
from typing import Optional, Dict, Any, List


class ExperimentTracker:
    """追踪每次实验运行，保存所有产物到结构化目录。

    目录结构:
        outputs/experiments/<experiment_name>/<run_id>/
            config.json       # 本次运行的参数快照
            metrics.jsonl     # 逐行 JSON 记录（可在运行中追加）
            summary.json      # 最终汇总
            ...               # 图表、模型等制品
    """

    def __init__(self, experiment_name: str, base_dir: str):
        self.experiment_name = experiment_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, "experiments",
                                    experiment_name, self.run_id)
        self.metrics: List[Dict[str, Any]] = []
        self.artifacts: List[str] = []
        os.makedirs(self.run_dir, exist_ok=True)

    def log_params(self, params: dict) -> None:
        """保存参数快照"""
        config_path = os.path.join(self.run_dir, 'config.json')
        # 过滤掉不可序列化的项
        serializable = {}
        for k, v in params.items():
            try:
                json.dumps(v)
                serializable[k] = v
            except (TypeError, ValueError):
                serializable[k] = str(v)
        with open(config_path, 'w') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """追加指标记录"""
        record = {'timestamp': datetime.now().isoformat()}
        if step is not None:
            record['step'] = step
        record.update(metrics)
        self.metrics.append(record)

        # 追加写入 JSONL
        metrics_path = os.path.join(self.run_dir, 'metrics.jsonl')
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def log_artifact(self, local_path: str) -> str:
        """将外部文件复制到运行目录"""
        import shutil
        dest = os.path.join(self.run_dir, os.path.basename(local_path))
        shutil.copy2(local_path, dest)
        self.artifacts.append(dest)
        return dest

    def log_figure(self, fig, name: str, dpi: int = 150) -> str:
        """保存 matplotlib Figure 到运行目录"""
        path = os.path.join(self.run_dir, f'{name}.png')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=dpi)
        self.artifacts.append(path)
        return path

    def save_summary(self) -> str:
        """保存最终汇总 JSON"""
        summary = {
            'experiment': self.experiment_name,
            'run_id': self.run_id,
            'metrics_count': len(self.metrics),
            'artifacts': [os.path.basename(a) for a in self.artifacts],
            'final_metrics': self.metrics[-1] if self.metrics else None,
        }
        summary_path = os.path.join(self.run_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary_path

    @property
    def save_dir(self) -> str:
        """便捷访问运行目录"""
        return self.run_dir
