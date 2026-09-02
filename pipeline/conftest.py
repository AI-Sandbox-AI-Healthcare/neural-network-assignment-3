"""
conftest.py — Assignment 3 integrity guard. DO NOT EDIT.

Runs once before any test is collected. It verifies that the PROVIDED parts of
pipeline.py are byte-for-byte as shipped:

    load_patient_visits, PAIN_KEYWORDS, _VISIT_FEATURE_COLS

If you changed any of them, the whole test run aborts with an error — revert
the change. Only the 6 functions you implement and get_sandbox_params() may be
edited.
"""
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

# SHA-256 of each provided helper's whitespace-normalised source, as shipped.
_PROVIDED_FUNCS = {
    "load_patient_visits": "8b77fb158dfb36e67497836aaea5a01ec5c8722c25883bbabd2d79d7b716245c",
}
_PROVIDED_DATA = {
    "PAIN_KEYWORDS": "19b79ce8480fb6ec6628b8aa9b8aef0009dd9018436ee77cb9aa217104df5e6b",
    "_VISIT_FEATURE_COLS": "150d6f875bfa7f36daae4445645527038a980da74166df9715c32762bce1a8a9",
}


def _norm(src: str) -> str:
    return "\n".join(ln.rstrip() for ln in src.replace("\r\n", "\n").split("\n")).strip()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def pytest_sessionstart(session):
    module_name = os.environ.get("PIPELINE_IMPL", "pipeline")
    try:
        module = __import__(module_name)
    except Exception as exc:  # noqa: BLE001
        pytest.exit(f"{module_name}.py failed to import: {exc!r}", returncode=1)

    changed = []
    for name, expected in _PROVIDED_FUNCS.items():
        fn = getattr(module, name, None)
        if fn is None:
            changed.append(f"{name} (missing)")
        elif _sha(_norm(inspect.getsource(fn))) != expected:
            changed.append(name)
    for name, expected in _PROVIDED_DATA.items():
        value = getattr(module, name, None)
        if value is None:
            changed.append(f"{name} (missing)")
        elif _sha(json.dumps(value)) != expected:
            changed.append(name)

    if changed:
        pytest.exit(
            "Provided helper(s) modified: " + ", ".join(changed) + ". "
            "Revert them — only the 6 functions you implement and "
            "get_sandbox_params() may change.",
            returncode=1,
        )
