import logging
import os
from datetime import datetime


def setup_logger(name: str, log_dir: str = "logs", level=logging.INFO) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fh = logging.FileHandler(f"{log_dir}/{today}_{name}.log")
        ch = logging.StreamHandler()

        fmt = logging.Formatter("%(asctime)s | %(name)-16s | %(levelname)-8s | %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
