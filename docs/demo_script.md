# CallGraph demo (5-7 minutes) — script and shot list

Setup before recording: `docker compose up -d falkordb`, Ollama running, `docker compose up -d` (or `make serve`), browser at http://localhost:8010.
Pick the call IDs from the Call viewer drop-down (do not rely on any ID written here).

| # | Time | Screen | Say |
|---|------|--------|-----|
| 1 | 0:00-0:30 | **Overview** | "Call centers record thousands of calls but review 1-2%. CallGraph turns audio into a queryable knowledge graph." Point at the KPI tiles, intent distribution and recent calls. |
| 2 | 0:30-1:30 | **Upload & process** | Drop in a short English call, click Process call, and narrate the stages as they tick: transcription, privacy redaction, intent, emotion, dynamics, compliance, summary, extraction, graph write. |
| 3 | 1:30-3:00 | **Calls** -> open a call | Play the audio; the transcript highlights in sync. Show speaker colours, per-turn emotion, intent, frustration, compliance checks, summary, dynamics, emotion timeline. |
| 4 | 3:00-3:45 | **Analytics** | Intent mix, emotion by intent, frustration by issue, talk ratio vs outcome, compliance pass rate, dynamics. Mention the number of calls behind them. |
| 5 | 3:45-4:30 | **Graph explorer** | Issues -> actions -> outcomes; search a node; click an issue to reveal the calls behind it. |
| 6 | 4:30-6:00 | **Ask CallGraph** | Click an example question with Graph selected, expand "Show generated Cypher", then ask the same question with Vector and Hybrid and compare answers and latency. |
| 7 | 6:00-6:45 | `outputs/eval/<run>/report.md` | Show the Vector vs Graph vs Hybrid table by question type, and the honest limits: small subsets, silver labels, small local models. |

Closing line: "It assists supervisors; it does not make decisions about agents."

Do not show UniData audio or transcripts publicly (licence CC BY-NC-ND); use SpokenWOZ or AxonData calls on screen.
