# Playback Viewer Model Owner Design

## Accepted behavior

Interactive playback must preserve its current entrypoints, MuJoCo XML/MJB loading,
resolved-visual-model fallback order, log messages, camera behavior, backend selection,
GLFW capability check, overlay rendering and monkeypatch seams.

## Confirmed defect

`playback_overlay.py` owns both overlay geometry and Viewer resource/model resolution.
`playback_viewer.py` consequently imports private model-loading and Viewer-launch helpers
from the overlay owner. This is a confirmed Divergent Change and reverse caller-knowledge
edge, not merely a line-count issue.

## Owner boundary

- `playback_viewer.py` owns Viewer composition, model resolution/loading, backend and GLFW
  launch decisions, and camera/focus setup helpers.
- `playback_overlay.py` owns MuJoCo debug geometry, motion/reward/velocity drawing, and
  overlay body selection.
- `scripts/play_interactive.py` remains a compatibility entrypoint and keeps importing the
  same Viewer-owned functions.

No new protocol, class, mutable state, schema, fallback, backend capability or public
configuration is introduced.

## Effect sketch

`scripts/play_interactive.py` -> `playback_viewer.py` -> environment scene artifacts / shared
render-play resolver / MuJoCo model -> interactive Viewer runtime.

`playback_viewer.py` -> `playback_overlay.py` only for drawing and overlay selection.

The forbidden edge after the change is `playback_viewer.py` -> private Viewer model-loading
or launch implementation in `playback_overlay.py`.

## Evidence

An architecture fitness test must fail while Viewer resource functions remain defined in
the overlay module. Existing visualization and script tests pin model resolver injection,
binary loading, fallback behavior and compatibility entrypoints. Offline evidence does not
claim live Viewer startup, simulation, training or policy quality.
