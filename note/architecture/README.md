# ICE-Cal Architecture pages

This directory owns the ICE-Cal Concept Figure and Design Inspector, together with the local runtime
needed to open them. The pages were initially migrated from FEMR, but their target-repository role is
now ICE-Cal architecture documentation.

The updated Design Inspector is the human projection for the v022 live-privileged grouped-DR source
teacher and the current axis-bank calibratable-Tracker design. The active source and Context
Contracts remain the complete semantic authority. Current implementation evidence is linked
separately through `atlas_manifest.json`; the figures themselves remain research-design artifacts
rather than runtime evidence.

## Open locally

The root HTML files embed their data and use local Rough.js/KaTeX assets, so they can be opened
directly without a server:

- `08_in_context_execution_calibration.html`
- `09_in_context_execution_calibration_design_inspector.html`

For HTTP serving or source-link inspection, double-click `open_atlas.command` or run:

```bash
./note/architecture/open_atlas.command
```

The launcher binds a loopback-only Node.js server to port `8767` and opens the page index.

## Local dependency closure

- Two ICE-Cal `concept/*.data.json` page models.
- The shared Atlas renderer and its statically imported layout modules.
- Local Rough.js and KaTeX packages, fonts, and upstream licenses.
- A loopback server, launcher, package manifests, and focused structural checks.

## Provenance

- Donor: `/Users/chengyuxuan/ArtiIntComVis/FEMR`
- Donor revision: `1cce99ff0c42ba70535b16d13df88d86c2ada258`
- Donor state: dirty; page 08 data and its checker had local modifications at migration time.
- Migration date: 2026-08-14
- License: Apache-2.0 in donor and target; Rough.js and KaTeX retain bundled licenses.
