# Palette NEAT SDK Demo (drone_seminar_neat)

Two USB webcams on a SiMa.ai Modalix DevKit, four AI workloads on the Modalix MLA, voice and
typed commands.

| Workload | What it does | Runtime | Runs on |
|---|---|---|---|
| YOLO26m Detection | LEFT camera, object detection | NEAT C++ `Model::Runner` | Modalix MLA |
| YOLO26m Segmentation | RIGHT camera, instance segmentation | NEAT C++ `Model::Runner` | Modalix MLA |
| Whisper-medium | speech to text (Korean / English) | pyneat 0.4.0 `genai.ASRModel` | Modalix MLA |
| Qwen3-0.6B | understands commands the parser cannot decide | pyneat 0.4.0 `genai.GenAIModel` | Modalix MLA |

The deterministic command parser is ordinary software logic. It is not an AI workload.

For a single MIPI camera with YOLO26 and live Neat Insight output, see the
[standalone MIPI object detector](mipi-detector/README.md).
The [SCOUT mission console](mipi-detector/SCOUT.md) adds Gemma 4 snapshot
inspection, live tracking and visual evidence using the same camera, with an
optional USB landing/payload watch view.

```
./run.sh              preflight, then start everything on the DevKit; open https://<devkit>:8022
./run.sh --stop-all   stop everything this project started
./run.sh --status     what is running
./run.sh --check      read-only environment check
./run.sh --selftest   offline checks
```

To deploy on another DevKit you need only this directory and two GenAI model directories
(`MODEL_ROOT/whisper-medium-a16w8`, `MODEL_ROOT/Qwen3-0.6B-Autoround-a16w4`). The YOLO archives
are inside the project (`assets/models/`).

Documentation:

- [docs/README.md](docs/README.md): architecture, commands, UI, configuration, measured results
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): copying the project to another DevKit
- [docs/SEMINAR_RUNBOOK.md](docs/SEMINAR_RUNBOOK.md): before the session, start, emergencies
- [docs/PORTABILITY_AUDIT.md](docs/PORTABILITY_AUDIT.md): self-containment audit

## Repository skills

Two Codex skills capture the SCOUT development and operating workflows:

- [scout-development](.agents/skills/scout-development/SKILL.md): camera pipelines,
  mission state, Gemma checks, evidence handling and the browser console.
- [scout-modalix-operations](.agents/skills/scout-modalix-operations/SKILL.md):
  launch, stop/restart, SoM/DVT configuration and stream/runtime troubleshooting.

Example prompts:

```text
Use $scout-development to add persistent mission evidence storage.
Use $scout-modalix-operations to diagnose why the USB view is offline on my board.
```

These skills live in `.agents/skills/` so they can be committed and shared with
the repository. Codex can select them for matching tasks or through an explicit
`$skill-name` mention. See the [official skill documentation](https://learn.chatgpt.com/docs/build-skills)
for discovery details; restart Codex if newly added skills do not appear.
