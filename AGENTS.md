# ICE-Cal Agent Principles (UniLab-derived runtime)

**Always use `uv run`, not python**.

UniLab 是一个 **高性能、模块化、contract 驱动** 的 RL infrastructure 仓库。

## 最小执行范围与停止条件

先区分“执行已明确的修改”和“排查原因未知的故障”。用户对本次任务指定的范围和验证方式优先。高速模式减少流程开销，不降低诊断深度，也不授权训练、仿真或扩大修改范围。

### 已明确的修改

1. 动手前，用一句话明确本次交付物和唯一必要验证。
2. 默认只运行一次针对性验证；已有直接证据足够时可以不运行。不得把多个测试套件打包成一条命令来规避这个限制。
3. 验证通过后立即交付，不追加测试、审计、文档或相邻清理。
4. 只有验证失败或发现具体阻塞时才能追加工作，并说明哪个事实失败、追加动作要解决什么。“更放心”“可能相关”“保持全面一致”不构成追加理由。
5. 不将维护所有相关旧测试作为默认完成条件。其他发现只报告，不顺手处理。
6. 收尾说明实际改了什么、验证了什么，以及是否超出原定范围。

### 原因未知的故障排查

1. 不套用“一次验证通过即交付”的停止条件。局部检查通过不等于故障已解释；允许继续进行有明确判别目的的只读代码检查、版本与配置对照，以及小型离线验证。用户明确禁止的检查仍不得执行。
2. 每次追加检查必须针对一个具体问题，说明不同结果分别支持或排除什么；完成后报告新增事实。没有新增信息时应改变调查方法，不重复阅读相同内容或重跑相同检查。
3. 从用户观察到的失败追踪到实际输入、状态变化和最终消费者。区分代码实现错误、设计规则问题和训练效果；不得将合理猜测写成根因，也不得未经证据就把原因限定为几个开关之间的选择。
4. 先用代码、已有配置和模型记录、小型离线对照缩小范围。不得用反复训练或回放代替排查，也不得因用户说训练快就默认安排 live test。确需 live test 时，先说明离线无法区分的具体问题、一次运行能够区分的结果和停止条件，并获得授权。
5. 停止条件是：找到有证据支持、能解释失败的原因；或已穷尽当前有判别价值的离线路径，明确指出缺失的具体事实及其阻断的结论。不能仅因达到检查次数、一个辅助检查通过或需要尽快交付就结束诊断；也不能为追求“全面”无限扩展。
6. 诊断不自动授权修复。汇报应给出发现、影响和有依据的处理建议；尚未定因时直说，不把一串排除项包装成已解决问题。

### 流程开销

按任务选择最少必要 skill，完整读取其必需指令，但不加载无关参考或反复读取本轮已掌握且未变化的材料。普通任务默认在对话中记录，不新增计划文档、审计报告或其他流程产物。时间应优先用于形成代码证据，而不是重复流程。

## Core Principles

1. **Contract first**: 不为了一次通过绕过 env / backend / runner contract。
2. **Fix at owner layer**: `scripts/` 只组装流程，不承载长期业务规则。
3. **Config first**: task / reward / backend 优先通过 Hydra + registry 表达。
4. **Backend isolation**: MuJoCo / Motrix 差异留在 backend 适配层和配置层。
5. **Evidence only**: support claim 只写仓库里已有的注册、配置、测试或 benchmark 事实。
6. **Validate near risk**: 在最接近风险的边界补验证，不只跑顶层命令。
7. **Cold-path asset access only**: asset/XML/model metadata 只允许在 init / materialization / cache 等低频路径处理；热路径不能解析 asset，也不能靠 `getattr` / `hasattr` 探测 backend 私有能力。
8. **Branch discipline**: 绝对不在未经用户明确请示/批准的情况下新建分支（`git branch` / `git switch -c` / `git checkout -b`）。确有需要时先说明理由并等待批准。

## High-Risk Areas

| 区域 | 不可破坏的不变量 |
|------|----------------|
| Env  | `NpEnvState.obs` 必须是 dict；`reset()` 返回 `(obs_dict, info_dict)`；`obs_groups_spec` 影响 wrapper 和 learner 维度。 |
| Config / Reward | reward 通过 Hydra 注入；后端切换必须通过 `task=<task>/<backend>` 选择 owner YAML，`training.sim_backend` 只是 owner YAML 的身份字段，不能单独 override 来切后端。算法超参数直接走 YAML compose，不经 Python 层解释。 |
| Backend | backend-specific 逻辑留在 backend / env 适配层，不向训练脚本扩散。env 层只能调用 `SimBackend`（`base.py`）中已声明的方法；若某方法只在 MuJoCo 或 Motrix 中存在，必须先将其加入 `SimBackend` 抽象接口（可抛 `NotImplementedError`），禁止直接在 env 里调用 backend 子类的私有方法（即"功能泄漏/feature leakage"）。新增 backend 专有能力时，需同步更新 `SimBackend`。 |
| Asset / Metadata | `ASSETS_ROOT_PATH`、`model_file`、XML / asset 元数据只允许在 init / materialization / cache 等低频路径访问；`step/reset/domain randomization` 等热路径不得解析 asset 或基于 asset 元数据做运行时分支。 |
| Asset / XML structure | `<keyframe>` 必须放在 task-level XML（`scene_*.xml` 或 `locomotion_task.xml` 等 fragment），**禁止放进 robot.xml**。robot.xml 是纯机器人描述（body / joint / actuator / sensor），跟 task / 场景无关；keyframe 是 task 起始姿态，属于场景或 task 资源。motrix 后端需要 keyframe 时通过 `scene.fragment_files` 引用 fragment XML。 |
| Async | 不绕开 runner lifecycle，也不另起 collector / learner 同步协议。 |
| Sim2Sim 契约 | 跨后端 play 时，影响策略 I/O / 网络结构的字段必须跨后端一致；不一致即 `CrossBackendIncompatibleError`。详见下方 Sim2Sim 章节。 |

## Sim2Sim 跨后端配置契约

`src/unilab/training/sim2sim.py` 按 dotted path 维护三类字段：

- **DENYLIST**（差异即 `CrossBackendIncompatibleError`）：`algo.obs_groups`、`env.control_config.action_scale`、`algo.policy.actor_hidden_dims` / `critic_hidden_dims`、`algo.empirical_normalization` / `algo.obs_normalization`、`env.sampling_mode`。`env.*` 子集对**任一方向**的不对称出现也 fail-closed；`algo` 专属字段目标缺省时按设计跳过（跨算法合法）。
- **WARNING_LIST**：`reward.*`、`env.control_config.simulate_action_latency`、`env.ctrl_dt`。
- **ALLOWLIST**（自由覆盖）：`training.sim_backend`、`env.scene`、`training.play_steps`、`env.domain_rand`、`env.noise_config`、`env.commands.vel_limit`。

训练时 `ExperimentTracker.start()` 把上述字段写入 `run_config.json` 的 `contract_snapshot`（不改 checkpoint 格式，旧 run 无 snapshot 时 fallback + warning）；五个 play 入口在建 env 前调用 `resolve_sim2sim_config` 校验，并用 `policy_load_dim_guard` 包裹 checkpoint 加载以把维度不匹配的隐晦报错重抛为显式诊断。设 `training.sim2sim_strict=false` 可把 DENYLIST 差异降级为 warning（默认 `true`）。DENYLIST 字段应通过 task 的 `base.yaml` 共享（范例：`conf/ppo/task/g1_walk_flat/{base,mujoco,motrix}.yaml`）；跨后端契约审计见 `scripts/audit_sim2sim_contracts.py`。

## Pointers

- PPO: `scripts/train_rsl_rl.py`
- MLX PPO: `scripts/train_mlx_ppo.py`
- APPO: `scripts/train_appo.py`
- SAC / TD3: `scripts/train_offpolicy.py`
- env contract: `src/unilab/base/np_env.py`
- backend contract: `src/unilab/base/backend/base.py`
- training run helpers: `src/unilab/training/run.py`
- visualization helpers: `src/unilab/visualization/`
- env shared numeric helpers: `src/unilab/envs/common/rotation.py`, `src/unilab/envs/common/math.py`
- MLX rotation helpers: `src/unilab/algos/mlx/common/rotation.py`
- config schema: `src/unilab/structured_configs.py`
- async runner: `src/unilab/ipc/async_runner.py`
- sim2sim 跨后端契约: `src/unilab/training/sim2sim.py`

## GitHub CLI (gh) 速查

### Issue 查看
```bash
gh issue view <number>
gh api repos/<owner>/<repo>/issues/<number> --jq '.body'
```

### PR 创建与管理
```bash
gh pr create --title "标题" --body "内容" --base main
gh pr list
gh pr view
```

### PR Gate

创建或更新 PR 前必须满足：

1. 最终提交已经完成，且 `git status --short --branch` 确认工作树干净。
2. 最终提交已经通过 `make test-all`。
3. 如果用户明确说明已经跑过 `make test-all`，不要重复跑；但必须在 PR body 的 Validation 里记录 `make test-all` 已完成。
4. 如果 `make test-all` 未通过且用户没有明确 override，不要创建或更新 PR。

### CI 工作流查看
```bash
gh run list
gh run list --workflow=<workflow-name>
gh run view <run-id>
gh run list --status=failure
```

### 常用组合
```bash
gh api repos/unilabsim/UniLab/issues/174 --jq '.title, .body'
git push -u origin fix/issue-174-mlx-ppo-config-alignment
gh pr create --title "fix: xxx" --body "Fixes #174" --base main
```

## Context

- ICE-Cal 文档与当前治理状态：[note/README.md](note/README.md)
- Concept Figure 与 Design Inspector：[note/architecture/README.md](note/architecture/README.md)
- 当前实现 Contract、证据与历史：[note/fada/README.md](note/fada/README.md)
- 开发者入口（环境、命令、提交规范）：[CONTRIBUTING.md](CONTRIBUTING.md)
- 稳定工程说明与操作手册：[docs/README.md](docs/README.md)
- 单卡服务器训练资源控制：[docs/runbooks/server-training.md](docs/runbooks/server-training.md)
- 通用 UniLab 框架文档不在本仓库维护；需要时查阅上游仓库。
