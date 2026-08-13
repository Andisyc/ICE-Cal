# Differentiable Context Run Rejected

- Branch at launch: `codex/context-differentiable-trajectory`
- Source checkpoint SHA-256: `d35a32d93b0387e534f6fcdd86b724c44187e308dbca1412435bffe95b6ed90c`
- Fault: left-knee actuator strength `0.7`
- Remote artifact: `/ssd1/cyx/liujun/FADA_ntr/artifacts/fada_context/run_left_knee_070_closed_loop_001`
- Completion: 10/10 rounds, 1792 final dataset rows, 64/64 valid gate rows per round
- Result: 0/10 Context candidates accepted by real MuJoCo
- Mean relative result: Context trajectory MSE worsened by `27.32%`
- Best candidate: worsened by `0.69%`
- Worst candidate: worsened by `50.97%`
- Final candidate: worsened by `34.97%`
- Safety behavior: every candidate Context and optimizer state was rolled back
- Final checkpoint: Context residual head is exactly zero; it has no adaptation effect

This is runtime-confirmed negative evidence. It proves the learned-dynamics training route failed for
this experiment; it does not prove every Context method is infeasible.
