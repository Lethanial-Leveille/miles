"""Pre-import runtime selection. Production keeps its existing entry point."""

import os
import sys

_runtime = None


def configure(runtime):
    global _runtime
    if _runtime is not None:
        raise RuntimeError("MILES runtime already configured")
    if runtime.mode not in {"dev", "test"} or not runtime.locked:
        raise RuntimeError("MILES runtime requires a supported mode and locked state")
    _runtime = runtime


def current():
    if _runtime is None and ("MILES_MODE" in os.environ or "pytest" in sys.modules):
        raise RuntimeError("Use the isolated MILES launcher: ./scripts/dev")
    return _runtime


def headless():
    runtime = current()
    return runtime is not None and runtime.mode in {"dev", "test"}


def development():
    runtime = current()
    return runtime is not None and runtime.mode == "dev"


def refuse_live(provider):
    if headless():
        raise FileNotFoundError(f"{provider} is disabled in the headless runtime")
