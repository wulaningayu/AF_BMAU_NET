# ============================================================
# logger.py
# Package-wide logging setup. Configures the "af_bmau_net" logger
# to write to both the console and a log file inside the current
# experiment's folder.
# ============================================================

import logging
import sys
from pathlib import Path


def setup_logger(log_dir, name="af_bmau_net", level=logging.INFO):
    """
    Configure the package logger to stream to stdout and to
    <log_dir>/train.log. Submodules should use
    logging.getLogger(__name__) and rely on propagation to this
    logger's handlers.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
