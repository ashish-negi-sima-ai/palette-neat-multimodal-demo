# SCOUT runtime troubleshooting

Use this reference for a diagnosed runtime issue. Current target evidence takes
precedence over the earlier measurements below. Full launch commands and measured
results live in [SCOUT.md](../../../../mipi-detector/SCOUT.md).

## Select the board profile

| Validated setup | SoM with Waveshare carrier | Earlier DVT |
|---|---|---|
| IMX678 camera name | `imx678 6-0042` (current overlays) | `imx678 0-0042` |
| Overlay | `modalix-som-waveshare-ECON-IMX678-1CAM.dtbo` | `modalix-dvt-ECON-IMX678-0CAM.dtbo` |
| Measured 1080p MIPI rate | 30 FPS | 25 FPS |
| Capture option | `--allow-cpu-fallback` was required | Strict capture worked on the validated DVT |
| Runtime | Matched Neat/LLiMa 0.4.0 packages | A feature build needed matching private LLiMa libraries |

The SoM used Type-A straight camera cables. Check the actual carrier, connector
and sensor registration before interpreting an overlay name. Do not copy the
DVT `SCOUT_LIBRARY_PATH` workaround to the SoM merely because both use Modalix.
Consult the DVT compatibility section in `SCOUT.md` only for that matching setup.

## Diagnose the failing stage

| Observation | Next evidence or action |
|---|---|
| SSH times out/refuses connection | Check the latest supplied IP and connectivity; do not change application code to address a transport failure. |
| Many video nodes but no `cam -l` camera | Inspect board identity, active overlay and sensor/I²C kernel messages. Sensor registration precedes application diagnosis. |
| Sensor rectangle/property warnings with a registered camera | Test a supported capture mode. These warnings alone did not prevent the validated IMX678 stream. |
| `Required zero-copy DMA-BUF pool is unavailable` | Use the explicit `--allow-cpu-fallback` compatibility path for the affected stack; YOLO still uses the accelerators. |
| Low FPS reported for a default sensor mode | Compare actual dimensions and capture timing. The SoM's default 4K mode reported about 8.76 FPS; explicit 1080p delivered 30 FPS. Raising `--fps` alone does not create a supported mode. |
| USB unavailable | Inspect `/dev/v4l/by-id/*-video-index0`, permissions, capture ownership and negotiated dimensions. The eMeet Nova was validated at 1280×720 MJPEG; `/dev/video97` was only its then-current node number. |
| Gemma model construction fails | Check the deployed config and both referenced ELF sets, then package compatibility and the selected Python environment. A directory's existence is not proof of a complete download. |
| Inference fails after a snapshot | Verify the complete model group stayed resident. Per-request model unload/reload is no longer supported. On model loss, stop all SCOUT owners and release their buffers before a fresh application start. |
| Camera watchdog or `IPI overflow recovery` during Gemma | Capture current diagnostics before changing scheduling. Earlier DVT failures led to a pause workaround; the current SoM passed continuous MIPI/USB capture with resident Gemma, Whisper Medium and Supertonic, including the browser voice-to-inspection flow. |
| CSI lanes not in stop state after a capture timeout | With SCOUT stopped, a scoped rebind of the actual IMX678 sensor recovered the 2026-09-28 SoM failure without rebooting. Confirm device identity and restore its binding in a finally block; do not apply this on every inspection. |
| Python import succeeds but native runtime crashes | Inspect versions and loaded libraries. Compare `pyneat`, Neat, GStreamer plugins and LLiMa as a release set; a library's ABI filename alone does not identify its package version. |

For native graph failures, capture `runtime.last_error()` and available structured
diagnostics (`error_code`, `repro_note`, terminal bus entries, reproduction launch)
before retrying. If a targeted fix leaves the same failure, collect the new logs
and narrow the failing stage instead of repeating identical restarts. Driver
rebinds, overlay edits, reboots and package replacement are not routine launch steps.

## Trace Insight delivery

1. Check the Insight service's `/api/health` and SDK port mapping. Default API
   port is 9900 and WebRTC signalling is 8081, but external mappings can differ.
2. Inspect `/api/ingest/stats`: MIPI uses `--channel`, USB uses `--channel + 1`.
   Video ports start at `--video-port-base`, metadata at `--metadata-port-base`.
   Confirm both point at the Insight host and the matching camera channel.
3. If ingest works, inspect `/api/egress/stats` and the browser's connection,
   decode and presentation statistics. SCOUT proxies `/offer` to Insight and
   renders through the shared `web/webrtc.js` client.
4. If video works but overlays do not, check exact timestamp matches and
   `arrivalFallbacks` in the browser. Forwarded metadata is not proof of
   correlation. MIPI's restart offset must be identical for video and metadata;
   USB must use the same receive PTS for both.
5. Refresh the browser after static UI edits. When testing a paused VLM request,
   verify subsequent frames and matched overlays, not just the retained image.

If installed, `sima-use-neat-insight` supplies further endpoint guidance. The
repository remains usable without that personal skill.
