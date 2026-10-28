"""Stage logging and observational RAM/CUDA monitoring for PMQ commands."""

from __future__ import annotations

import json
import platform
from importlib.metadata import version, PackageNotFoundError
from contextlib import contextmanager
from contextvars import ContextVar
import uuid
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
    tree_rss = snapshot["rss_bytes"]
    try:
        children = process.children(recursive=True)
    except (psutil.Error, OSError) as error:
        children = []
        snapshot["ram_sampling_error"] = repr(error)
    for child in children:
        try:
            tree_rss += child.memory_info().rss
        except (psutil.Error, OSError) as error:
            snapshot["ram_sampling_error"] = repr(error)
    snapshot["process_tree_rss_bytes"] = tree_rss
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
        self.run_id = uuid.uuid4().hex
        self.components = {}
        self.active_component = None
        self.component_rss = 0
        self.max_tree_rss = 0
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
            if value.get("ram_sampling_error"):
                self.sampling_error = value["ram_sampling_error"]
            value["run_id"] = self.run_id
            value["component"] = self.active_component
            self.stream.write(json.dumps(value) + "\n")
            self.stream.flush()
            self.samples += 1
            rss = value.get("process_tree_rss_bytes", value["rss_bytes"])
            self.max_tree_rss = max(self.max_tree_rss, rss)
            if self.active_component is not None:
                self.component_rss = max(self.component_rss, rss)
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
        summary = self.directory / f"{self.stage}_resources.json"
        previous_id = json.loads(summary.read_text()).get("run_id", uuid.uuid4().hex) if summary.exists() else uuid.uuid4().hex
        for suffix in (".log", "_resources.jsonl", "_resources.json"):
            path = self.directory / f"{self.stage}{suffix}"
            if path.exists():
                archive = path.with_name(path.stem + "." + previous_id + path.suffix)
                if archive.exists():
                    raise FileExistsError(archive)
                path.rename(archive)
        self.environment = _environment()
        self.start = time.perf_counter()
        self.token = _current_monitor.set(self)
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
        self._write_progress()
        self.thread = threading.Thread(target=self._loop, name="pmq-resources", daemon=True)
        self.thread.start()
        return self

    def _totals(self):
        core_names = {"comp_stats", "solve_lp", "quant_gptq"}
        core_wall = sum(v["wall_seconds"] for k, v in self.components.items() if k in core_names)
        peaks, core_peaks = {}, {}
        for name, record in self.components.items():
            for device, value in record["per_gpu_peak_memory_gib"].items():
                peaks[device] = max(peaks.get(device, 0), value)
                if name in core_names:
                    core_peaks[device] = max(core_peaks.get(device, 0), value)
        return {"core_wall_seconds": core_wall,
                "core_time_percent": {k: 100 * v["wall_seconds"] / core_wall for k, v in self.components.items() if k in core_names and core_wall},
                "per_gpu_peak_memory_gib": peaks,
                "peak_gpu_memory_sum_gib": sum(peaks.values()),
                "core_peak_gpu_memory_sum_gib": sum(core_peaks.values()),
                "max_component_peak_gpu_memory_sum_gib": max((v["peak_gpu_memory_sum_gib"] for v in self.components.values()), default=0.0)}

    def _write_progress(self):
        payload = {"schema_version": 2, "run_id": self.run_id, "stage": self.stage,
                   "status": "running", "components": self.components,
                   "active_component": self.active_component,
                   "wall_seconds": time.perf_counter() - self.start,
                   "measurement": _measurement(), **self._totals()}
        path = self.directory / f"{self.stage}_resources.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)

    @contextmanager
    def component(self, name):
        if self.active_component is not None:
            raise RuntimeError("Resource components must not overlap")
        self.active_component = name
        self.component_rss = 0
        reset_error = None
        try:
            _cuda_peaks(reset=True)
        except Exception as error:
            reset_error = repr(error)
        self._write_progress()
        started = time.perf_counter()
        status, error_text = "executed", None
        try:
            yield
        except BaseException as error:
            status, error_text = "failed", repr(error)
            raise
        finally:
            peaks, measurement_error = {}, reset_error
            try:
                peaks = _cuda_peaks()
            except Exception as error:
                measurement_error = repr(error)
            elapsed = time.perf_counter() - started
            # No nvidia-smi subprocess here: that belongs to the sampling thread.
            try:
                process = psutil.Process()
                self.component_rss = max(self.component_rss, process.memory_info().rss)
            except (psutil.Error, OSError) as error:
                self.sampling_error = repr(error)
            record = {"status": status, "error": error_text,
                      "measurement_error": measurement_error, "wall_seconds": elapsed,
                      "per_gpu_peak_memory_gib": peaks, "peak_gpu_memory_sum_gib": sum(peaks.values()),
                      "max_sampled_process_tree_rss_bytes": self.component_rss}
            existing = self.components.get(name)
            if existing:
                record["wall_seconds"] += existing["wall_seconds"]
                record["per_gpu_peak_memory_gib"] = {k: max(v, existing["per_gpu_peak_memory_gib"].get(k, 0)) for k, v in {**existing["per_gpu_peak_memory_gib"], **peaks}.items()}
                record["peak_gpu_memory_sum_gib"] = sum(record["per_gpu_peak_memory_gib"].values())
                record["max_sampled_process_tree_rss_bytes"] = max(record["max_sampled_process_tree_rss_bytes"], existing["max_sampled_process_tree_rss_bytes"])
                if existing["status"] == "failed":
                    record["status"], record["error"] = "failed", existing.get("error")
            self.components[name] = record
            self.active_component = None
            self._write_progress()
            LOGGER.info("COMPONENT_COMPLETE name=%s %s", name, json.dumps(record))

    def __exit__(self, error_type, error, traceback):
        _current_monitor.reset(self.token)
        self.stop.set()
        self.thread.join()
        self._sample()
        self.stream.close()
        summary = {
            "stage": self.stage, "status": "failed" if error else "complete",
            "schema_version": 2, "run_id": self.run_id, "pid": os.getpid(), "command": sys.argv,
            "components": self.components, "measurement": _measurement(),
            "max_sampled_process_tree_rss_bytes": self.max_tree_rss,
            **self._totals(),
            "wall_seconds": time.perf_counter() - self.start,
            "sample_interval_seconds": self.interval,
            "environment": self.environment,
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
        path = self.directory / f"{self.stage}_resources.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(path)
        for handler in self.handlers:
            LOGGER.removeHandler(handler)
            handler.close()
        LOGGER.setLevel(self.old_level)
        LOGGER.propagate = self.old_propagate
        sys.stdout, sys.stderr = self.old_stdout, self.old_stderr
        self.console_log.close()
        return False


_current_monitor = ContextVar("pmq_run_monitor", default=None)


def _measurement():
    return {"wall_seconds": "CUDA-synchronized component elapsed time",
            "per_gpu_peak_memory_gib": "torch.cuda.max_memory_reserved / 2**30; excludes non-PyTorch allocations",
            "peak_gpu_memory_sum_gib": "sum of per-GPU peaks; peaks need not coincide; component maximum sum also reported separately",
            "max_sampled_gpu_global_used_mib": "sampled whole-device usage, may include other processes; never used as allocator peak"}


def _cuda_peaks(reset=False):
    if not torch.cuda.is_initialized():
        return {}
    ids = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    if not all(value.strip().isdigit() for value in ids):
        ids = list(range(torch.cuda.device_count()))
    peaks = {}
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
        if reset:
            torch.cuda.reset_peak_memory_stats(device)
        else:
            peaks[str(ids[device])] = torch.cuda.max_memory_reserved(device) / 2**30
    return peaks


@contextmanager
def component(name):
    monitor = _current_monitor.get()
    if monitor is None:
        yield
    else:
        with monitor.component(name):
            yield


def _environment():
    result = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
              "python": platform.python_version(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpus": []}
    try:
        result["transformers"] = version("transformers")
    except PackageNotFoundError:
        result["transformers"] = None
    try:
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_properties(index)
                result["gpus"].append({"logical_device": index, "name": device.name,
                                       "total_memory_bytes": device.total_memory})
    except Exception as error:
        result["gpu_query_error"] = repr(error)
    return result
