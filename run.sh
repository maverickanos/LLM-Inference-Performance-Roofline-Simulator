#!/usr/bin/env bash
# llm-roofline entry points: ./run.sh {demo|test|report [args...]}
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

case "${1:-demo}" in
  demo)
    "$PY" -m llm_roofline demo
    ;;
  test)
    "$PY" -m pytest tests/ -q
    ;;
  report)
    shift
    if [ "$#" -eq 0 ]; then
      "$PY" -m llm_roofline report --model llama-3-70b --gpu h100 --tp 4 \
        --batch 1..256 --context 8192 --out results
    else
      "$PY" -m llm_roofline report "$@"
    fi
    ;;
  *)
    echo "usage: ./run.sh {demo|test|report [--model ... --gpu ...]}" >&2
    exit 1
    ;;
esac
