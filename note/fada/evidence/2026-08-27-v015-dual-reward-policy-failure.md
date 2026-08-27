# v015 Dual-Reward Policy Failure

Status: current invalidation evidence for v015; not v016 policy evidence.

During the v015 privileged-Oracle run, the Electerm training dashboard showed:

- around iteration 1340: mean Reward `-79.6`, episode length `46.6`, termination `100%`;
- around iteration 1540: mean Reward `-75.2`, episode length `31.7`, termination `100%`;
- around iteration 2340: mean Reward `-48.1`, episode length `23.1`, termination `100%`.

The reported return became less negative while episodes shortened and termination remained complete.
This falsifies the intended ordering “survive and recover is better than terminate early” and is
consistent with early-termination Reward hacking. The run does not prove that every possible dual
Reward formulation fails; it proves the active v015 objective is inadmissible and may not be tuned or
reused as the v016 baseline.
