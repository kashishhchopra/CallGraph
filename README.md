# CallGraph — call-center intelligence platform

Raw call audio → speaker-labelled transcript → six analysis modules (intent, emotion, dynamics, compliance, summary, entity
extraction) → FalkorDB knowledge graph + transcript vector index → GraphRAG chat and a Streamlit dashboard.
Academic, non-commercial project. **Nothing numeric in this README is a result**; results are generated into
`outputs/eval/<run_id>/report.md` by `make eval`.

## Setup
```bash
make setup                 # venv, deps, spaCy model, FalkorDB container, Ollama model
cp .env.example .env       # then fill HF_TOKEN (never commit .env)
```
`.env` keys: `HF_TOKEN` (pyannote, gated), `LLM_API_KEY` (only for OpenAI/Anthropic providers), and optional overrides using
`__` nesting, e.g. `FALKORDB__HOST`, `API__PORT`, `LLM__MODEL`. Environment variables override `configs/*.yaml`.

Gated pyannote models require clicking *Agree* (same HF account as the token) on:
`pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`, `pyannote/wespeaker-voxceleb-resnet34-LM`,
`pyannote/speaker-diarization-community-1`. Without access the pipeline logs a warning and falls back to MFCC clustering
(recorded in each call's `metadata.diarization_backend`). Stereo recordings with independent channels are transcribed
per channel (no diarization needed).

## Run each phase
| Phase | Command |
|---|---|
| Download datasets (subsets from `configs/datasets.yaml`) | `make download` |
| Manifests | `make manifests` |
| Train | `make train-intent train-emotion train-summary` |
| Process calls → graph | `make process` |
| Evaluate + report | `make eval` |
| Serve API + web UI (http://localhost:8010) | `make serve` or `docker compose up --build` |
| Tests / lint | `make test lint` |

CallHome (TalkBank) is manual: register at https://talkbank.org, accept the CallHome English terms, put media in
`data/raw/callhome/audio` and CHAT files in `data/raw/callhome/cha`, then `make manifests`.

## Configuration, no hard-coding
Paths, models, hyper-parameters, thresholds, label maps, rubric, graph schema, datasets, UI pages/charts, RAG settings and the
analytics Cypher all live in `configs/`; prompts live in `prompts/*.j2`. `tests/test_no_hardcoding.py` fails on magic numbers
and hard-coded URLs/model ids in `src/`, `api/`, `dashboard/`.

* **Add a dataset:** add an entry to `configs/datasets.yaml` (with a `source:` block), write a `DatasetLoader` subclass decorated
  `@DATASETS.register("name")` in `src/callgraph/data/loaders/`, and import it in that package's `__init__`.
* **Add an analysis module:** subclass `AnalysisModule`, decorate `@MODULES.register("name")`, add it to `configs/pipeline.yaml`.
* **Add a chart:** add a Cypher metric in `configs/analytics.yaml` and a chart entry in `configs/dashboard.yaml`.

## Licences and data handling
Datasets are for academic, non-commercial use. `data/raw`, `data/processed`, model weights and `outputs/` are git-ignored; never
publish UniData files or derivatives (CC BY-NC-ND). AxonData is CC BY-NC. Download scripts fetch from official sources
(TweetSumm uses the community mirror `Andyrasika/TweetSumm-tuned` because the official release needs Kaggle's `twcs.csv`).
PII (phones, emails, cards, names) is redacted with Presidio before anything is stored.

## Honest scope notes
* The public UniData sample contains 5 calls and AxonData 11 (7 languages); real-call results are therefore small-sample.
* "Real-call" emotion/compliance/intent/extraction labels are **LLM-annotated silver labels, not human gold labels**
  (two independent LLM annotators, tie-break pass, Cohen's kappa reported in `data/labels/*.json`).
* GraphRAG questions are templated; reference answers are produced by executing Cypher on the processed-call graph.
* Models are small (Whisper base, DistilBERT, wav2vec2-base, local 3B LLM) because the development machine has 8 GB RAM and no GPU.
* UniData's `Category` is an industry, not an intent, so intent is compared with silver intent labels instead.

## Dataset citations
SpokenWOZ (Si et al., 2023); AMI (Carletta et al., 2005); Earnings-22 (Del Rio et al., 2022); VoxConverse (Chung et al., 2020);
CallHome (TalkBank); ABCD (Chen et al., 2021); Banking77 (Casanueva et al., 2020); CREMA-D (Cao et al., 2014);
RAVDESS (Livingstone & Russo, 2018); MELD (Poria et al., 2019); DialogSum (Chen et al., 2021); TweetSumm (Feigenblat et al., 2021);
UniData call-center sample (UniDataPro, HF); AxonData call-center sample (HF). Methods: Whisper (Radford et al., 2022),
pyannote.audio (Bredin et al.), GraphRAG (Edge et al., 2024), RAGAS (Es et al., 2023). Verify each against its official page.

## Results and label files
* Full evaluation report (generated, every number read from saved files): `outputs/eval/<run_id>/report.md`
  (tables as CSV/JSON, charts as PNG, `case_studies/` per-call reports, `human_rating_sheet.csv` for the 30-summary rating study).
  Rebuild with `make eval` (runs every evaluator) or `make report` (re-renders from saved files).
* **LLM-annotated silver labels, not human gold labels** (two annotator models, a tie-break pass, Cohen's kappa recorded in each file):
  `data/labels/real_call_emotion.json`, `real_call_compliance.json`, `real_call_intent.json`, `extraction_check.json`.
* GraphRAG question set (templated questions, reference answers produced by executing Cypher on the graph): `data/labels/graphrag_questions.json`.
* Demo script and shot list: `docs/demo_script.md`.

## Known limitations
* Real-time streaming mode (`features.streaming`) is a config flag only; it is not implemented. No React UI.
* Audio emotion is a frozen wav2vec2 encoder with a trained classifier head, not a fully fine-tuned model.
* Evaluations use small subsets (see each table); real-call results rest on 16 calls (7 English) and low-agreement silver labels.
* CallHome needs manual TalkBank registration and was not run.
