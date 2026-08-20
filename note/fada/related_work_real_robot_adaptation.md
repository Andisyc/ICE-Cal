# Related Work: 真机微调与仿真训练

> 用途：为 ICE-Cal / FADA 的 Supervisor 展示和后续会话提供可检索的论文分类。
>
> 分类标准：只有真实机器人 rollout、reward、人工 correction 或真实输入输出轨迹实际进入参数/模型更新，才归入“真机微调/校准”。真机只做部署评估的工作归入“仿真训练”。

## A. 真机微调 / 真机校准

| 方法 | 机器人 | 真机监督或梯度来源 | 更新对象 | 方法定位 |
|---|---|---|---|---|
| [FADA](https://arxiv.org/abs/2606.28476) | Unitree G1、Booster T1 | 约 2 分钟 observation-action rollout；监督学习，无 reward | IDM 内 LoRA | 局部执行模块校准 |
| [Robot Trains Robot (RTR)](https://arxiv.org/abs/2508.12252) | Humanoid | 真机速度跟踪 reward，PPO；机械臂提供保护、反馈和 reset | dynamics latent 与真机 Critic | 真机强化学习校准 |
| [SLowRL](https://arxiv.org/abs/2603.17092) | Unitree Go2 | 真机 reward，PPO；recovery policy 安全接管 | Actor、Critic 的 rank-1 LoRA | 低秩真机策略微调 |
| [Legged Robots that Keep on Learning](https://arxiv.org/abs/2110.05457) | Unitree A1 | 真机交互 reward、自动 reset、持续 RL | 完整 locomotion policy | 完整策略真机微调 |
| [TRANSIC](https://arxiv.org/abs/2405.10315) | 机械臂 | 人类在真机执行时介入，提供 correction action | residual policy | 人工纠正驱动的残差校准 |
| [ASAP](https://arxiv.org/abs/2502.01143) | Unitree G1 | 真机轨迹监督 delta action model | residual model；策略随后在校准后的仿真中微调 | real-to-sim-to-real 间接微调 |
| [SPI-Active](https://arxiv.org/abs/2505.14266) | Legged robot | 主动采集真机轨迹，提高参数可辨识性 | simulator / physical parameters | 主动系统辨识 |
| [HALO](https://arxiv.org/abs/2603.15084) | Humanoid | 真机输入输出轨迹 | nominal robot model 与 payload 参数 | 真机数据驱动的系统辨识 |
| [Sim-and-Real Co-Training](https://arxiv.org/abs/2503.24361) | 机械臂、Humanoid | 真实数据与仿真数据混合监督 | 整体策略 | 混合域联合训练 |
| [Actuator Reality Shaping](https://arxiv.org/abs/2607.02205) | 多类机器人 | 硬件响应测量；不进行任务策略梯度更新 | 真实执行器控制接口 | 硬件侧校准，不是策略微调 |

## B. 仿真训练，真机不更新参数

| 方法 | 仿真中学习的内容 | 真机阶段 |
|---|---|---|
| [RMA](https://arxiv.org/abs/2107.04034) | 域随机化下训练 base policy 与 adaptation module | 零样本部署，不微调 |
| [SplitAdapter](https://arxiv.org/abs/2606.03297) | 训练 load/context encoder 与 factorized adapter | G1 真机测试，不收集数据更新 |
| [PolySim](https://arxiv.org/abs/2510.01708) | 多仿真器联合训练，降低 simulator bias | G1 零样本部署 |
| [Joint Torque Perturbation](https://arxiv.org/abs/2603.21853) | 注入状态相关力矩扰动，覆盖复杂动力学误差 | 真机只验证鲁棒性 |
| [ADAPT](https://arxiv.org/abs/2606.16542) | 将解析 disturbance observer 输入策略 | 真机只执行与评估 |
| [ReactiveBFM](https://arxiv.org/abs/2606.30362) | 闭环 motion generation 与异步 replanning | G1 真机验证 |
| [Mind Your Steps](https://arxiv.org/abs/2606.08253) | 精确 foothold tracking policy | 真机验证落脚精度 |
| [Motion Generation + Motion Tracking](https://arxiv.org/abs/2604.17335) | generator 与 tracker 闭环训练 | 真机部署评估 |
| [RLPF](https://arxiv.org/abs/2506.12769) | 用仿真 physics feedback 微调 motion generator | 真机部署；feedback 不是来自真机 |
| [REFINE-DP](https://arxiv.org/abs/2603.13707) | 仿真 RL 联合微调 Planner 与低层 Controller | 真机只评估 |
| [HERO](https://arxiv.org/abs/2602.16705) | residual-aware EE tracker、IK reference 与 replanning | 真机只评估 |
| [RGB](https://arxiv.org/abs/2606.25123) | RL-guided MPPI 在线修正 tracking drift | 公开证据主要为 MuJoCo |
| [PhyGile](https://arxiv.org/abs/2603.19305) | 修复 motion generation 与 physical tracking mismatch | 主要为仿真训练 |
| [HALOMI](https://arxiv.org/abs/2606.18772) | controller-aware reference adaptation | 真机部署，不属于目标域动力学校准 |

## C. FADA 引用的真机适配路线

| FADA 引用 | 对应路线 |
|---|---|
| Smith et al., 2021 | 真机完整策略 RL 微调 |
| Hu et al., 2025 / RTR | 真机 dynamics latent 优化 |
| Jiang et al., 2024 / TRANSIC | 真机人工纠正与 residual learning |
| He et al., 2025 / ASAP | 真机 residual dynamics/action learning |
| Sobanbabu et al., 2025 / SPI-Active | 真机系统辨识 |
| Maddukuri et al., 2025 | 仿真与真机数据联合训练 |
| Lei et al., 2026 | sim-and-real co-training 机制分析 |

## D. 核心结论

| 路线 | 核心做法 | 与 FADA 的边界 |
|---|---|---|
| 仿真训练 | 在域随机化、多仿真器或扰动中学习鲁棒性/适应性，真机不更新 | RMA、SplitAdapter、PolySim 等 |
| 真机模型校准 | 用真机轨迹拟合 simulator、residual 或物理参数 | HALO、SPI-Active、ASAP |
| 真机策略微调 | 用真机 reward、人工纠正或混合数据更新策略 | RTR、SLowRL、Keep on Learning、TRANSIC |
| FADA | 冻结 Planner，只用少量无 reward 真机 rollout 监督微调 IDM | 将更新范围限制在 factorized execution interface |

## 证据边界

上述分类基于论文摘要、项目页和 FADA 的 related-work 叙述。对于“真机只评估”类论文，结论是未发现真机数据进入训练，而不是断言作者绝对没有内部真机数据。
