# Engineering document scope

Engineering documents explain stable repository structure, ownership, public interfaces, and
runtime dataflow. They must describe existing production code and link to its owner; they must not
duplicate research semantics from an active Contract.

Use this directory for subjects such as:

- training and playback entrypoint maps;
- observation, action, backend, async, and checkpoint ownership;
- stable producer-to-consumer runtime flows;
- compatibility and persistence boundaries.

Use [`../runbooks/`](../runbooks/) for executable operating procedures. Use
[`../../note/`](../../note/) for Concept Figures, Design Inspectors, Contracts, plans, reviews, and
evidence. Until a dedicated engineering page exists, the current code-level entrypoint map remains
in [`AGENTS.md`](../../AGENTS.md#pointers).

