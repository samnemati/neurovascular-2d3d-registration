#!/usr/bin/env bash
# Runs the whole study: prepare the cases, sweep registration, compare target
# realism, then write the tables and figures. Uses the project venv if present.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}"
"$PY" scripts/10_prepare_cases.py
"$PY" scripts/20_run_registration.py
"$PY" scripts/25_target_realism.py
"$PY" scripts/30_figures.py
