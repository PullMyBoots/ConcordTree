# Application source update — 2026-11-08

This snapshot exports the current validated portable application directory into `app/`, preserving the Linux/WSL launchers, model weights, required native extensions, Web dependencies and third-party notices. It excludes Python/scientific environments, input datasets, local jobs, caches and research workspace state.

The application now supports configurable AP-NJ/refinement pipelines, evolutionary models within likelihood settings, pause/download/resume, and removal of unused experimental methods. The local interface includes CPU-thread/GPU selection fixed per job, a persistent NVIDIA monitor, and separate Geometry, Nature and seasonal weather themes with a draggable ten-minute cycle.

The right panel is an algorithm-stage strip with the running stage highlighted. Its table changes with the selected stage: AP-NJ has time only; distance, MLP, likelihood and kernel refinement have relevant rounds, time, criterion values and moves. Completed stages can be reviewed while the running stage remains highlighted. The redundant log title/status labels and irrelevant AP-NJ columns are removed.

## Validation

- The current app manifest verifies all 1,134 delivered files before publication.
- The source and portable Web adapters each passed 42 API/host-monitor tests, including resource snapshot validation, GPU streaming/reconnection, and structured progress events.
- Browser checks covered settings persistence, actual resource selection, current-stage highlighting, per-stage columns, disabled/reordered modules, previous-stage review, no wrapping/clipping, and a saved real job.
- One gene example and one species example produced byte-identical annotated and unannotated trees before/after the monitoring/dashboard changes, on the same L40S with GPU 1 and four CPU threads. These small regression checks do not establish general speed or accuracy equivalence for every custom pipeline.
- Continuous live sampling retained all three installed GPUs under one persistent monitor process; stale telemetry and failed UI requests retained identity/readings and recovered automatically.

Evidence is included in [app/evidence/RUN_DASHBOARD.json](app/evidence/RUN_DASHBOARD.json), [app/evidence/STAGE_PANEL.json](app/evidence/STAGE_PANEL.json), [app/evidence/MODULAR_SETTINGS.json](app/evidence/MODULAR_SETTINGS.json), and [app/evidence/SEASON_WEATHER.json](app/evidence/SEASON_WEATHER.json). Technical-report results retain their own configurations and provenance.

## Artifact boundary

This is a source push. No binary archive/container was rebuilt, and no GitHub Release/tag was created. Earlier distribution validation, installers, restore bundles and build scripts remain historical records; they must not be presented as newly tested artifacts for this updated source. The source checkout does not contain the complete runtime needed to launch the packaged application.
