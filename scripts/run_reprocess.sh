#!/bin/sh
# Re-process all calls after the role-assignment fix, then rebuild everything that depends on them.
cd "$(dirname "$0")/.."
PY="env PYTHONPATH=. .venv/bin/python"; L=outputs/logs
$PY scripts/process.py unidata --force > $L/re_unidata.log 2>&1
$PY scripts/process.py axondata --force > $L/re_axon.log 2>&1
$PY scripts/process.py spokenwoz --limit 40 --force > $L/re_swoz.log 2>&1
$PY scripts/build_questions.py > $L/re_questions.log 2>&1
$PY scripts/silver_labels.py extraction > $L/re_silver_extraction.log 2>&1
$PY -m eval.run roles intent_audio compliance real_world dynamics_vs_outcome rag > $L/re_eval.log 2>&1
$PY -m eval.report > $L/re_report.log 2>&1
echo DONE > $L/re_all.done
