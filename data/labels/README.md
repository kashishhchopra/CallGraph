# Label files

Generated locally (not committed, because they contain text derived from UniData, licence CC BY-NC-ND):

* `real_call_emotion.json`, `real_call_compliance.json`, `real_call_intent.json`, `extraction_check.json` —
  **LLM-annotated silver labels, not human gold labels** (two annotator models, tie-break pass, Cohen's kappa inside each file).
  Rebuild: `python scripts/silver_labels.py emotion|compliance|intent|extraction` after `make process`.
* `graphrag_questions.json` — templated GraphRAG questions with reference answers produced by executing Cypher on the graph.
  Rebuild: `PYTHONPATH=. python scripts/build_questions.py`.
