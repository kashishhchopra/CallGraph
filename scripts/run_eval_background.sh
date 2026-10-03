#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=outputs/logs
while [ ! -f $L/all.done ]; do sleep 30; done
$PY scripts/dataset_cli.py manifests > $L/manifests_final.log 2>&1
$PY -m eval.run > $L/eval.log 2>&1
$PY -m eval.report > $L/report.log 2>&1
echo DONE > $L/eval.done
