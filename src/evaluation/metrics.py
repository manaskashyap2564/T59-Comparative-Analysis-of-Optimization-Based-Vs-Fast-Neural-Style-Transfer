"""
Evaluation & Benchmark Module — StyleSense
Logs stylization run metrics to CSV.
Owner: Manas Kashyap (+ Shubhansh for model-side timing)
"""

import csv
import os
import time
from datetime import datetime


LOG_FILE = "logs/benchmark_results.csv"
FIELDNAMES = [
    "run_id", "timestamp", "method", "resolution",
    "runtime_ms", "style_loss", "content_loss", "notes"
]


def init_log():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        print(f"Log initialized: {LOG_FILE}")


def log_run(method: str, resolution: int, runtime_ms: float,
            style_loss: float, content_loss: float, notes: str = ""):
    """Appends one stylization run to the benchmark CSV."""
    init_log()
    run_id = f"run_{int(time.time())}"
    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "resolution": resolution,
        "runtime_ms": round(runtime_ms, 2),
        "style_loss": round(style_loss, 6),
        "content_loss": round(content_loss, 6),
        "notes": notes
    }
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
    print(f"Logged: {run_id} | {method} | {resolution}px | {runtime_ms:.1f}ms")
    return run_id


if __name__ == "__main__":
    # Smoke test
    log_run("fast", 512, 118.5, 0.0421, 0.0113, "test run")
    log_run("optimization", 512, 4200.0, 0.0198, 0.0089, "test run")
    print("Metrics module working.")
