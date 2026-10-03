#!/bin/sh
# Sequential long-running steps (one at a time to fit 8 GB RAM). Each step is idempotent/cached.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=outputs/logs; mkdir -p $L
[ -f models/emotion_audio/labels.json ] && [ -z "$RETRAIN_AUDIO" ] || $PY scripts/train_emotion.py audio > $L/train_emo_audio.log 2>&1
$PY scripts/tune_fusion.py > $L/tune_fusion.log 2>&1
[ -f models/summarization/config.json ] || $PY scripts/train_summary.py > $L/train_summary.log 2>&1
$PY scripts/process.py axondata > $L/proc_axon.log 2>&1
$PY scripts/process.py spokenwoz --limit 40 > $L/proc_swoz.log 2>&1
for t in emotion compliance intent extraction; do $PY scripts/silver_labels.py $t > $L/silver_$t.log 2>&1; done
$PY scripts/build_questions.py > $L/questions.log 2>&1
echo DONE > $L/all.done
