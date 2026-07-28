import atexit
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
from collections import defaultdict


def _get_git_commit() -> str | None:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# Pricing tables are loaded at import time from pricing/*.json siblings of
# this file. Drop a new pricing/<anything>.json file to extend without
# editing cost_tracker.py — files without "_kind" or with "_kind":"llm" go
# into MODEL_PRICING_PER_MILLION (values: [input_per_M, output_per_M]).
# For Anthropic extended thinking, output_tokens includes thinking tokens
# (billed at output rate). Long-context (>1M token) Opus pricing is
# approximate.
MODEL_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {}


def _load_pricing_sidecars() -> None:
    # A missing pricing/ dir is tolerated at import time; add_llm_api_cost asserts
    # the table is non-empty.
    pricing_dir = Path(__file__).parent / "pricing"
    if not pricing_dir.is_dir():
        return
    for jf in sorted(pricing_dir.glob("*.json")):
        data = json.loads(jf.read_text())
        kind = data.pop("_kind", "llm")
        data.pop("_comment", None)
        if kind == "llm":
            for model, pair in data.items():
                if model.startswith("_"):
                    continue
                MODEL_PRICING_PER_MILLION[model] = (float(pair[0]), float(pair[1]))
        else:
            raise ValueError(f"Unknown pricing _kind={kind!r} in {jf}")


_load_pricing_sidecars()

# Modal pricing (per second) — as of Feb 2026, https://modal.com/pricing
MODAL_CPU_PER_CORE_PER_SEC = {
    "sandbox": 0.00003942,
    "standard": 0.0000131,
}
MODAL_MEM_PER_GIB_PER_SEC = {
    "sandbox": 0.00000672,
    "standard": 0.00000222,
}
MODAL_GPU_PER_SEC = {
    "B200": 0.001736,
    "H200": 0.001261,
    "H100": 0.001097,
    "A100-80GB": 0.000694,
    "A100": 0.000583,  # 40GB
    "L40S": 0.000542,
    "A10G": 0.000306,
    "L4": 0.000222,
    "T4": 0.000164,
}


class ModalTimer:
    """Yielded by track_modal context managers. Set .elapsed to override wall-clock timing."""
    elapsed: float | None = None


class CostTracker:
    def __init__(self, cost_file: Path, run_description: str | None = None, **save_kwargs):
        self.cost_file = cost_file
        self.cost_file.parent.mkdir(parents=True, exist_ok=True)
        self.run_description = run_description
        self.save_kwargs = save_kwargs
        self.run_cost = 0.0
        self.start_time = datetime.now(timezone.utc)
        self.git_commit = _get_git_commit()
        self.warned = set()
        self.run_cost_by_model = defaultdict(float)
        self.run_cost_by_model_input_output = defaultdict(lambda: defaultdict(float))
        self.modal_compute_cost = 0.0
        self.prior_cumulative_cost = self._load_cumulative_cost()
        atexit.register(self._save_on_exit)

    def _load_cumulative_cost(self) -> float:
        """Read total_cost.jsonl and sum all run_cost values."""
        if not self.cost_file.exists():
            self.cost_file.touch()
            return 0.0
        cumulative = 0.0
        with open(self.cost_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line in cost file: {line}")
                        continue
                    cumulative += entry.get("run_cost", 0.0)
        return cumulative

    def add_llm_api_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Add cost for an API call. Returns cost for this call."""
        assert MODEL_PRICING_PER_MILLION, (
            "No LLM pricing loaded — copy the pricing/ directory alongside cost_tracker.py"
        )
        if model not in MODEL_PRICING_PER_MILLION and model not in self.warned:
            print(f"Warning: Model {model} not in pricing list. Using default Sonnet pricing.")
            self.warned.add(model)
        input_price, output_price = MODEL_PRICING_PER_MILLION.get(model, (3.0, 15.0))  # Default to Sonnet pricing
        input_cost = input_tokens * input_price / 1_000_000
        output_cost = output_tokens * output_price / 1_000_000
        cost = input_cost + output_cost
        self.run_cost_by_model[model] += cost
        self.run_cost_by_model_input_output[model]["input"] += input_cost
        self.run_cost_by_model_input_output[model]["output"] += output_cost
        self.add_cost(cost)
        return cost

    def add_modal_gpu_cost(
        self, wall_seconds: float, gpu: str, gpu_count: int = 1, cpu: float = 0, memory_mib: int = 0, is_sandbox: bool = True,
    ) -> float:
        """Add cost for a Modal GPU run. CPU/memory default to 0 since GPU cost dominates."""
        return self.add_modal_cost(wall_seconds, cpu=cpu, memory_mib=memory_mib, is_sandbox=is_sandbox, gpu=gpu, gpu_count=gpu_count)

    def add_modal_cost(
        self, wall_seconds: float, cpu: float, memory_mib: int, is_sandbox: bool = True, gpu: str | None = None, gpu_count: int = 1,
    ) -> float:
        """Add cost for a Modal run. Returns the computed cost.

        gpu: GPU type string matching MODAL_GPU_PER_SEC keys (e.g. "H100", "H200", "B200").
        gpu_count: number of GPUs (default 1, ignored if gpu is None).
        """
        tier = "sandbox" if is_sandbox else "standard"
        cost = (
            cpu * wall_seconds * MODAL_CPU_PER_CORE_PER_SEC[tier]
            + (memory_mib / 1024) * wall_seconds * MODAL_MEM_PER_GIB_PER_SEC[tier]
        )
        if gpu is not None:
            assert gpu.upper() in MODAL_GPU_PER_SEC, f"Unknown GPU type {gpu!r}. Known: {list(MODAL_GPU_PER_SEC)}"
            cost += gpu_count * wall_seconds * MODAL_GPU_PER_SEC[gpu.upper()]
        self.modal_compute_cost += cost
        self.add_cost(cost)
        return cost

    @contextmanager
    def track_modal_gpu(self, gpu: str, gpu_count: int = 1, cpu: float = 0, memory_mib: int = 0, is_sandbox: bool = True):
        """Context manager for Modal GPU runs. CPU/memory default to 0 since GPU cost dominates."""
        with self.track_modal(cpu=cpu, memory_mib=memory_mib, is_sandbox=is_sandbox, gpu=gpu, gpu_count=gpu_count) as t:
            yield t

    @contextmanager
    def track_modal(self, cpu: float, memory_mib: int, is_sandbox: bool = True, gpu: str | None = None, gpu_count: int = 1):
        """Context manager that times a block and records Modal compute cost.

        Yields a ModalTimer. Set t.elapsed = <seconds> from inside the container
        for accurate timing (excludes scheduling wait). Falls back to wall-clock if not set.
        """
        t = ModalTimer()
        start = time.monotonic()
        try:
            yield t
        finally:
            elapsed = t.elapsed if t.elapsed is not None else (time.monotonic() - start)
            self.add_modal_cost(elapsed, cpu=cpu, memory_mib=memory_mib, is_sandbox=is_sandbox, gpu=gpu, gpu_count=gpu_count)

    def add_cost(self, cost: float):
        # could add hook here if desired
        self.run_cost += cost

    def total_cost(self) -> float:
        return self._load_cumulative_cost() + self.run_cost

    def get_token_usage_budget(self, default: float = 5000.0) -> float:
        budget_file = self.cost_file.parent / ".token_usage_budget"
        if not budget_file.exists():
            return default
        try:
            return float(budget_file.read_text().strip())
        except ValueError:
            print(f"Warning: .token_usage_budget exists but couldn't parse as float. Using default {default}.")
            return default

    def get_api_usage_budget(self, default: float = 5000.0) -> float:
        budget_file = self.cost_file.parent / ".api_usage_budget"
        if not budget_file.exists():
            return default
        try:
            return float(budget_file.read_text().strip())
        except ValueError:
            print(f"Warning: .api_usage_budget exists but couldn't parse as float. Using default {default}.")
            return default

    def is_over_budget(self) -> bool:
        budget = self.get_api_usage_budget()
        total = self.total_cost()
        return total > budget


    def _save_on_exit(self):
        """Append run summary to cost file on exit."""
        if self.run_cost == 0:
            return  # Don't log runs with no API calls
        entry = {
            "start_timestamp": self.start_time.isoformat(),
            "end_timestamp": datetime.now(timezone.utc).isoformat(),
            "script": sys.argv[0],
            "full_args": sys.argv,
            "git_commit": self.git_commit,
            "description": self.run_description,
            **self.save_kwargs,
            "run_cost": self.run_cost,
            "modal_compute_cost": self.modal_compute_cost,
            "run_cost_by_model": self.run_cost_by_model,
            "run_cost_by_model_input_output": self.run_cost_by_model_input_output,
        }
        with open(self.cost_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
