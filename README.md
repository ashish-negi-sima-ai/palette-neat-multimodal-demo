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
