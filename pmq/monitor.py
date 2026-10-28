"""Stage logging and observational RAM/CUDA monitoring for PMQ commands."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import subprocess
import threading
import time
from pathlib import Path

import psutil
import torch


LOGGER = logging.getLogger("pmq")


def resource_snapshot() -> dict:
    process = psutil.Process()
    snapshot = {
        "time": time.time(),
        "pid": os.getpid(),
        "rss_bytes": process.memory_info().rss,
        "available_bytes": psutil.virtual_memory().available,
        "system_swap_used_bytes": psutil.swap_memory().used,
        "gpu": [],
        "cuda_process_memory": [],
    }
    if torch.cuda.is_initialized():
        snapshot["cuda_process_memory"] = [
            {"logical_device": index, "allocated_bytes": torch.cuda.memory_allocated(index),
             "reserved_bytes": torch.cuda.memory_reserved(index)}
            for index in range(torch.cuda.device_count())
        ]
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=True,
            )
            for line in result.stdout.splitlines():
                index, used, total, utilization = [value.strip() for value in line.split(",")]
                snapshot["gpu"].append({"physical_device": int(index), "used_mib": float(used),
                                        "total_mib": float(total), "utilization_percent": float(utilization)})
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            snapshot["gpu_query_error"] = str(error)
    return snapshot


def progress(stage: str, current: int, total: int, **fields) -> None:
    LOGGER.info("%s %s/%s %s", stage, current, total, json.dumps(fields, sort_keys=True))


class _Tee:
    def __init__(self, original, stream):
        self.original, self.stream = original, stream

    def write(self, value):
        self.stream.write(value)
        self.stream.flush()
        return self.original.write(value)

    def flush(self):
        self.stream.flush()
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


class RunMonitor:
    """Record resource samples without changing limits or stopping processes."""

    def __init__(self, directory: str | Path, stage: str, interval: float = 5.0):
        if interval <= 0:
            raise ValueError("monitor interval must be positive")
        self.directory, self.stage, self.interval = Path(directory), stage, float(interval)
        self.stop = threading.Event()
        self.samples = 0
        self.max_rss = 0
        self.min_available = None
        self.gpu_peaks = {}
        self.sampling_error = None
        self.gpu_query_errors = 0

    def _sample(self):
        try:
            value = resource_snapshot()
            self.stream.write(json.dumps(value) + "\n")
            self.stream.flush()
            self.samples += 1
            self.max_rss = max(self.max_rss, value["rss_bytes"])
            self.min_available = value["available_bytes"] if self.min_available is None else min(self.min_available, value["available_bytes"])
            if value.get("gpu_query_error"):
                self.gpu_query_errors += 1
                if self.gpu_query_errors == 1:
                    LOGGER.warning("GPU query failed: %s", value["gpu_query_error"])
            for gpu in value["gpu"]:
                index = str(gpu["physical_device"])
                self.gpu_peaks[index] = max(self.gpu_peaks.get(index, 0), gpu["used_mib"])
        except Exception as error:
            self.sampling_error = str(error)
            LOGGER.warning("resource sampling failed: %s", error)

    def _loop(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.start = time.perf_counter()
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.console_log = (self.directory / f"{self.stage}.log").open("a")
        sys.stdout = _Tee(self.old_stdout, self.console_log)
        sys.stderr = _Tee(self.old_stderr, self.console_log)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        self.handlers = [logging.StreamHandler(self.old_stderr), logging.FileHandler(self.directory / f"{self.stage}.log")]
        self.old_level, self.old_propagate = LOGGER.level, LOGGER.propagate
        for handler in self.handlers:
            handler.setFormatter(formatter)
            LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)
        LOGGER.propagate = False
        self.stream = (self.directory / f"{self.stage}_resources.jsonl").open("a")
        LOGGER.info("STAGE_START stage=%s pid=%s", self.stage, os.getpid())
        self._sample()
        self.thread = threading.Thread(target=self._loop, name="pmq-resources", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, error_type, error, traceback):
        self.stop.set()
        self.thread.join()
        self._sample()
        self.stream.close()
        summary = {
            "stage": self.stage, "status": "failed" if error else "complete",
            "wall_seconds": time.perf_counter() - self.start,
            "samples": self.samples, "max_sampled_process_rss_bytes": self.max_rss,
            "min_sampled_system_available_bytes": self.min_available,
            "max_sampled_gpu_global_used_mib": self.gpu_peaks,
            "sampling_error": self.sampling_error, "gpu_query_errors": self.gpu_query_errors,
        }
        if error:
            summary["error"] = repr(error)
            LOGGER.error("STAGE_FAILED stage=%s", self.stage, exc_info=(error_type, error, traceback))
        else:
            LOGGER.info("STAGE_COMPLETE %s", json.dumps(summary))
        (self.directory / f"{self.stage}_resources.json").write_text(json.dumps(summary, indent=2) + "\n")
        for handler in self.handlers:
            LOGGER.removeHandler(handler)
            handler.close()
        LOGGER.setLevel(self.old_level)
        LOGGER.propagate = self.old_propagate
        sys.stdout, sys.stderr = self.old_stdout, self.old_stderr
        self.console_log.close()
        return False
