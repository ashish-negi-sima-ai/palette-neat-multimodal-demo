# Portability and self-containment audit - Palette NEAT SDK Demo (2026-09-14)

The only allowed external, project-specific dependencies are `MODEL_ROOT/whisper-medium-a16w8`
and `MODEL_ROOT/Qwen3-0.6B-Autoround-a16w4`. Everything else must be inside this directory or
be part of the installed SiMa SDK / BSP / OS.

**Result: no runtime dependency on the project it was copied from or on any other user-created
workspace project.**

Re-run at any time with `python3 scripts/portability_audit.py` (read-only). The report of the
final audit is `runtime/test/audit/portability_audit.{txt,json}`.

## Static audit (whole project, `runtime/` included, 103 text files)

| Check | Result |
|---|---|
| Symlinks | none |
| Absolute paths into any of the 27 other `/workspace` directories | **none** |
| Absolute paths compiled into the vision binary (`strings`) | none |
| `/media` paths | `MODEL_ROOT` default in `config/default.env` only |
| Mentions of another project's name | provenance comments and one C++ namespace in `vision-local/src/` (unchanged from the source, not a path); the audit script's own documentation; this document and `docs/README.md` (provenance) |
| Stray bytecode | none; `run.sh` sets `PYTHONPYCACHEPREFIX=runtime/pycache` |
| Runtime data copied from the source project | none; `runtime/` was created empty and holds only this project's own logs and test evidence |

## Models (`config/model_manifest.json`)

| Model | Root | Path | Runs on |
|---|---|---|---|
| YOLO26m | PROJECT | `assets/models/yolo26m.tar.gz` | Modalix MLA |
| YOLO26m-seg | PROJECT | `assets/models/yolo26m-seg.tar.gz` | Modalix MLA |
| Whisper-medium | MODEL_ROOT | `whisper-medium-a16w8` | Modalix MLA |
| Qwen3-0.6B | MODEL_ROOT | `Qwen3-0.6B-Autoround-a16w4` | Modalix MLA |

## Runtime evidence

Every run on the DevKit (functional runs 1 and 2, the 32 min stability run and the cold-start run
after a reboot) was started as `/workspace/drone_seminar_neat/run.sh`. The logs show:

- the vision binary, both YOLO archives, `config/`, `backend/`, `voice/` and `web/` resolved inside this
  project;
- Whisper-medium and Qwen3-0.6B under `/media/nvme/llima/models`;
- every log, pid file, selection file, TLS certificate, Python cache and temporary file under this
  project's `runtime/`.

**Files the SiMa SDK writes outside the project** (fixed paths inside the SDK libraries):

- `/tmp/sima_gst_plugin_scanner_<pid>` (removed by `run.sh` on stop; verified none left after every Ctrl+C);
- `/tmp/sima_gst_registry_*` and `/tmp/rpmsg_lock_*`;
- `/tmp/neat_boxdecode_segment_contract.txt`;
- the model unpack cache under `/media/nvme/simaai/coprocessing/models/`.

## Deployment needs

- this directory;
- `MODEL_ROOT/whisper-medium-a16w8` and `MODEL_ROOT/Qwen3-0.6B-Autoround-a16w4`;
- the normal installed SiMa SDK / runtime (see `docs/DEPLOYMENT.md`).
