import psutil
import os
import gc
import logging

logger = logging.getLogger(__name__)

def log_memory(stage: str = ""):
    """Logs current RAM usage of the process."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"💾 MEMORY USAGE {f'[@ {stage}]' if stage else ''}: {mem_mb:.1f} MB")
    return mem_mb

def clear_memory(stage: str = ""):
    """Explicitly triggers garbage collection and logs memory."""
    before = log_memory(f"BEFORE {stage}" if stage else "BEFORE GC")
    gc.collect()
    after = log_memory(f"AFTER {stage}" if stage else "AFTER GC")
    logger.info(f"🧹 GC CLEARED: {before - after:.1f} MB")
