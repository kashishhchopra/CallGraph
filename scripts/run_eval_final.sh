#!/bin/sh
# Final re-runs: dynamics, roles (with production-pipeline accuracy), uncached RAG latency; then the report.
cd "$(dirname "$0")/.."
PY="env PYTHONPATH=. .venv/bin/python"; L=outputs/logs
$PY -m eval.run roles dynamics > $L/eval_final1.log 2>&1
env LLM__CACHE_ENABLED=false $PY -m eval.run rag_latency > $L/eval_final2.log 2>&1
$PY -m eval.report > $L/report2.log 2>&1
echo DONE > $L/eval_final.done
