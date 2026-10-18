"""日志配置：同时输出到文件（logging/ 目录）和控制台。

日志文件以启动时间命名，自动只保留最近的若干个日志文件。
"""

import logging
from datetime import datetime
from pathlib import Path

_DATE_FORMAT = '%Y_%m_%d__%H_%M_%S'
_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'


def setup_logging(log_path='logging', level=logging.INFO, keep=50):
    """配置根日志记录器，返回 logger。

    log_path: 日志文件目录；
    keep: 保留最近的日志文件数量，其余按修改时间从旧到新删除。
    """
    path = Path(log_path)
    path.mkdir(parents=True, exist_ok=True)

    _cleanup_old_logs(path, keep)

    log_filename = path / f'{datetime.now().strftime(_DATE_FORMAT)}.log'

    # force=True：重复调用时移除旧 handler 重新配置，保证生效
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger()


def _cleanup_old_logs(path, keep):
    """只保留最近 keep 个 .log 文件（按修改时间排序，删除最旧的）。"""
    log_files = sorted(path.glob('*.log'),
                       key=lambda p: p.stat().st_mtime)
    for old_file in log_files[:-keep] if keep > 0 else log_files:
        old_file.unlink()
