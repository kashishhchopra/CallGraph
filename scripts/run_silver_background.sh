#!/bin/sh
cd "$(dirname "$0")/.."
PY="env PYTHONPATH=. .venv/bin/python"; L=outputs/logs
for t in emotion compliance intent extraction; do $PY scripts/silver_labels.py $t > $L/silver_$t.log 2>&1; done
$PY scripts/build_questions.py > $L/questions.log 2>&1
echo DONE > $L/all.done
