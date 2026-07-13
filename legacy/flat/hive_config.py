import os

# OPT-P2: All tuneable constants are env-overridable — no magic numbers in nodes.
MODEL_NAME   = os.environ.get("HIVE_MODEL",        "qwen2.5-coder:7b")
MAX_RETRIES  = int(os.environ.get("HIVE_MAX_RETRIES", "3"))

# How long human_gate_node waits for operator acknowledgement before
# routing to retrospector automatically (headless / CI fallback).
GATE_TIMEOUT_SEC = int(os.environ.get("HIVE_GATE_TIMEOUT", "120"))

# SEC-C2: Minimal sandbox environment — no host secrets exposed to
# generated code running in reviewer_node's subprocess.
SAFE_ENV = {
    "PATH":       "/usr/bin:/bin:/usr/local/bin",
    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    "HOME":       "/tmp/sandbox",
}
