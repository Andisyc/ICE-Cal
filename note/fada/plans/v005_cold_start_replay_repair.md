# v005 Cold-start and Replay Repair

Terminal outcome: implement the accepted paper-aligned v005 source/replay contract without changing
the Planner or IDM network architecture and without starting formal training.

Main execution unit: add exact reset-aligned standing cold-start collection, persist row provenance,
separate IDM and Planner replay admission, stratify every Planner batch, and seal scenario-resolved
quality metrics in the production checkpoint serializer.

Embedded checks: OFF behavior, exact source windows, artifact round-trip, parent fail-closed identity,
sampling quotas, intermediate exclusion, ordered gradient owners, finite serializer metrics, focused
FADA regression, lint, Atlas validation, and one bounded real MuJoCo source sentinel.

Engineering acceptance: local code and document checks pass. Formal training and closed-loop policy
quality remain separate S4 boundaries.

Conditional escalation: do not train if a real source sentinel cannot produce all required strata or
if any artifact/checkpoint identity is missing or non-finite.
