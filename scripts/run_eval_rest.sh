#!/bin/sh
# Re-run the evaluators that did not complete (summarization, real_world, rag), then build the report.
cd "$(dirname "$0")/.."
PY="env PYTHONPATH=. .venv/bin/python"; L=outputs/logs
$PY -m eval.run summarization real_world rag dynamics_vs_outcome > $L/eval_rest.log 2>&1
$PY -m eval.report > $L/report.log 2>&1
echo DONE > $L/eval_rest.done
