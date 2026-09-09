# 指令门控相位接触调度 Oracle 设计（v024）

2026-09-07 讨论定稿。目标：解决 Oracle 走直线训练中 Gait Reward 与
Standing Reward 的冲突（策略轻微漂移以收割 Gait Reward 的 reward hacking），
用"显式双脚相位输入 + 指令门控"替代传统硬编码相位 Gait Reward。

治理状态：用户已确认语义方向并要求统一文档；`FADA-METHOD-v024` 与
`FADA-TRAIN-v024` 是当前 Contract。工程实现与离线 Module Test 已完成；正式运行链、
训练和策略质量仍未获本文件证明。v023 已转入历史，
`mujoco_fada_phase` 只保留为兼容与对照入口。

## 1. 问题陈述

- 走直线需要步态塑形（抬脚高度、触地时序），传统做法是 Gait Reward。
- Gait Reward 只问"脚的动作符不符合节拍"，不问"指令要不要你走"。
  零指令下交替抬脚/小步蹭地仍有正收益 → 漂移是理性最优。
- 与 Standing Reward 构成两套互相矛盾的目标函数，调权重只能在某个
  指令区间内妥协，永远存在一个 regime 其中一个被 hack。

## 2. v1–v5 失败的诊断（为什么不构成对思路的证伪）

历次实现每条关键性质都差了一点，见 git 历史 `514534aa`..`f0871532`：

| 缺陷 | 位置 | 后果 |
|---|---|---|
| 相位冻结挂在未开启的 `gait_constraint.enabled` 上，时钟对零指令 env 也以 1.5 Hz 空转 | `walk_control_bindings.py:262-274`，`walk_config.py:87` | 站→走起步相位随机，每次起步是不同的一题；相位观测语义被稀释 |
| v1 目标曲线 `((1+sin)/2)²` 全周期非零，支撑窗口为零；幅度 `0.06·d/0.3` 无饱和（1 m/s 时 0.2 m） | `walk_reward_bindings.py:490-491` | 奖励要求双脚几乎全程离地，行走本身被罚 |
| v3 改为正 exp 奖励，零指令 cost=0 白拿 +1 | `walk_reward_bindings.py:509` | critic 学到"站着全局最优"，速度跟踪被牺牲（后续 `tracking_lin_vel_sigma=0.04` 收紧即证据） |
| v4/v5 相对高度基准 `min(left_z, right_z)` 可被双脚腾空（hopping）白嫖；Bezier 曲线双支撑窗口近似为零；`min_forward_speed_for_gait_reward=0.0` 使门控退化为恒 1 | `walk_reward_bindings.py:521-531` | 奖励失去"一脚必须撑地"语义，支撑脚被持续往上推 |
| feet_phase 权重从 UniLab 的 5.0 降到 1.0 | `mujoco_fada_privileged_oracle_command_phase_grouped_dr_lineage.yaml` | 跟钟梯度与 tracking 梯度互相拉扯，落入拖脚/原地踏步局部最优 |

## 3. 成熟仓库调研（2026-09-07，三个仓库代码级证据）

### 3.1 死区：一致在指令生成端做

| 仓库 | 机制 | 阈值 |
|---|---|---|
| unitreerobotics/unitree_rl_gym | `_resample_commands` 中 xy 范数 ≤ 阈值置零 | 0.2 m/s |
| roboterax/humanoid-gym (XBot-L) | 同上 | 0.2 m/s |
| nvidia-isaac/WBC-AGILE | `min_vel_norm` 配置项，小指令精确归零 | 0.1 m/s |

要点：置零必须是**精确的 0**，下游才能用 `is_null_cmd = (command == 0).all()`
做离散门控。"零"从区间变成可检测的离散状态。

### 3.2 零指令行为：两个阵营

- **阵营 A（不站，原地踏步）**：Unitree G1、humanoid-gym。固定周期相位钟
  （0.8 s / 0.64 s），永不冻结、不触地重置，步态奖励不做指令门控，
  零指令下策略被奖励原地踏步。放弃了静止站立。
- **阵营 B（精确零门控 + 站立奖励组）**：WBC-AGILE。
  `aestetic_rewards.py` 中 `stand_with_both_feet_if_null_cmd`(w=10)、
  `equal_foot_force_if_null_cmd`(1)、`flat_orientation_if_null_cmd`(-4)、
  `relax_if_null_cmd` 全部挂在 `is_null_cmd` 精确零检测上；非零指令时
  全部不生效。其发布的 velocity 任务无相位步态奖励。
- **关键发现**：WBC-AGILE 存在未接线的
  `UniformVelocityGaitBaseHeightCommand`（`agile/rl_env/mdp/commands/commands.py`
  L530-560）：采样 1–2 Hz 的 sin/cos 相位时钟，**站立/零指令环境时钟频率
  冻结为 0**。"冻结"设计在工业代码中存在先例，但无已发布任务使用。

### 3.3 无先例声明

没有成熟仓库同时验证过本设计的完整组合：精确死区、离散 null/non-null
dispatcher、确定性相位状态机、双通道指令强度、速度相关步频和接触调度。
成熟先例只支持其中的死区、null-command 门控、固定相位钟或接触匹配等局部机制。
因此外部先例只用于约束结构，不能替代本仓库的模块测试和策略质量证据。

## 4. 设计决策（已确认）

### D1 指令死区归一化

- 指令采样/resample 后：`norm(cmd_xy) < 0.1` 且 `|cmd_yaw| < 0.1` → 精确置零。
- 与现有 30% 站立 env 采样叠加：先死区归零，再按比例强制零指令。
- 下游一切门控使用精确零检测 `is_null_cmd`，禁止区间判断。
- v024 首个配置固定 `heading_command=false`。若未来启用逐步 heading feedback，必须让
  feedback 后的最终 command 经过同一个归一化 owner；禁止在 Reward 或 observation
  消费端复制第二套死区。

### D2 指令 regime 与相位状态机

- 状态机逐 env 保存上一拍的 null/non-null regime，并拥有 reset 与 command 切换：
  - null reset / null→null：输出 stand phase `[π, π]`，不推进；
  - non-null reset / null→non-null：原子初始化为确定性的 π 相差起步相位；
  - non-null→non-null：保持 π 相差并按 D4 的 `f(s_cmd)` 推进；
  - non-null→null：吸附到 `[π, π]`，由接触 cost 引导悬空脚落地。
- 确定性起步相位必须满足“两脚起步时都在 stance 窗、固定同一领先腿、左右相差 π”。
  `[0, π]` 是工程初值，不是冻结常数；随机 `offset_phase` 只作为后续鲁棒性候选。
- 这套 command-transition 初始化不是 D6 的运行期 touchdown phase reset。
- phase 状态机直接由 `is_null_cmd` 拥有，不得再挂 `gait_constraint.enabled`。
- null command 直接把两脚接触目标规定为 stance，与站立天然一致，
  **不需要独立的 Standing Reward regime owner**。
- 保留两个与步态无关的站立质量项（同样 null-cmd 门控，参照 WBC-AGILE）：
  双足底力均分（修正重心分布）、力矩放松（修正能耗）。相位项不管这两件事。

### D3 接触调度奖励替换高度跟踪

- phase 的 owner 单位是弧度。每脚先计算
  `phase_fraction = (phase mod 2π) / (2π)`，再用
  `stance_expected = phase_fraction < duty_factor`；`duty_factor` 初值为 `0.55`。
- `contact_measured = foot_force_z > contact_force_threshold`；`1 N` 只是等待
  G1 MuJoCo 公共传感器标定的工程候选，不是冻结真值。
- 每脚 mismatch 只包含两种错误：摆动窗内触地，或支撑窗内没有触地。
  `contact_cost` 是两脚 mismatch 的均值，`reward_contact = -w_contact * contact_cost`。
  正确匹配得到 0，错误匹配得到有限负值；禁止 null command 白拿常数正奖励。
- 这是有界、可补偿的软约束，使开环钟失步后仍可恢复。可补偿不代表关键行为可
  反向排序；D5 的测试 regime 内必须满足排序卡。
- 不跟踪高度曲线（等式约束改不等式约束，Reward 不再把脚往上推）。
- 放弃所有相对高度基准（`min(foot_z)` 类），只允许绝对地面参考或接触力。
- 二值接触匹配本身不证明步态质量；轻触、滑擦、阈值抖动必须由足端 clearance、
  滑移速度、接触持续时间、支撑力和机身运动等外部量检查。

### D4 双通道指令强度与分段步频

- 死区之后，用平移和 yaw 的死区外强度共同构造 `s_cmd ∈ [0,1]`：
  `s_cmd = clip(max(excess(norm(cmd_xy))/linear_span,
  excess(abs(cmd_yaw))/yaw_span), 0, 1)`。两个 span 为正的配置参数；纯旋转、侧移、
  后退和前进都必须能激活相位钟。
- `excess(x)` 分别定义为 `max(x-dead_zone, 0)`；恰好等于 dead-zone 边界的
  command 属于 non-null，取 `s_cmd=0` 和 `gait_frequency=f_min`。
- null command：`gait_frequency = 0`。
- non-null command：
  `gait_frequency = f_min + (f_max - f_min) * s_cmd`。
- `f_min > 0`，因此死区边界处允许从 0 跳到 `f_min`；这是同一个离散 regime
  边界的语义，不允许增加连续的“几乎站立”第三 regime。
- 消除"小速度指令被迫 1.5 Hz 原地踏步"的结构性 hacking。
- `f_min` 与 `f_max` 是工程参数。固定 π 相差且 `duty_factor=d>0.5` 时，最长连续
  双支撑为 `(d-0.5)/f`，单脚支撑为 `(1-d)/f`；`duty_factor` 与 `f_min` 必须联合
  校准，不能假设降低频率只会改善起步。

### D5 Reward 职责与行为排序

- 不迁移 UniLab `5.0:1.0` 为先验，也不预设接触项必须大于 tracking。
- 速度 tracking 决定机身是否执行 command；接触调度排除错误支撑/摆动；两个
  null-command quality term 只改善载荷分配与能耗。
- contact mismatch 在解析形式上可补偿；但权重可行区必须使六条
  `FADA-V024-PHASE-CONTACT-ORDERING` 关系在指定测试 regime 内全部成立。
- fall penalty 只提供稠密软信号；`max_tilt_deg`、`min_base_height` 等 env
  termination 才拥有不可补偿的跌倒边界。
- 系数只能在公式符号、边界、truth table 和行为排序固定后校准；总 return
  不能代替逐项排序和外部行为量。

### D6 开环钟失步的缓解（分阶段）

- 本阶段（Oracle 平地、低扰动）：D3 只保证每步失步 cost 有界、对称；它不证明
  策略会自行恢复。恢复由模块 truth table 和有界策略实验分别验证。
- 若 `[0, π]`、固定 `duty_factor=0.55` 的起步失败，最小 fallback 是增加一个
  固定时长的双支撑 `STARTING` 状态；不得把它伪装成调权修复。
- 鲁棒性阶段候选（暂不实现）：
  - 触地相位重置：检测到足跟着地时将该脚相位 snap 到支撑相起点；
  - 策略可调步频：相位增量作为动作分量，钳制标称频率 ±50%，
    加频率偏离惩罚（CPG-RL 路线，复杂度高）。

## 5. 预期行为（验收基准）

### 5.1 模块不变量

1. 死区内 command 与精确零 command 产生相同的 null regime、`[π,π]` phase 和
   downstream observation/Reward 输入。
2. null→non-null 从确定性双支撑相位恢复 π 相差；持续 non-null 永不丢失相差；
   non-null→null 回到 `[π,π]`。
3. 纯 yaw、侧移、后退和前进 command 都产生非零 `s_cmd` 和有界步频。
4. phase/contact truth table 使用 cycle fraction 和 `duty_factor=0.55`，正确匹配
   cost 为 0，任一 mismatch 为有限正 cost；Reward 取其负值。

### 5.2 Reward 行为排序

1. 零命令稳定双脚站立 > 零命令原地交替踏步。
2. 低速正确前进 > 低速命令下静止。
3. 低速正确前进 > 原地交替踏步。
4. 低速正确前进 > 拖脚滑行。
5. 纯旋转正确步态 > 双脚冻结后扭转身体。
6. 行走转站立后受控落脚并稳定站立 > 长期单脚悬空、单脚站立或失稳。

这些关系是部分序，不声明未列行为之间的全序。策略验收还必须分别报告命令跟踪、
足端位移/步长、clearance、滑移速度、接触持续时间、左右支撑力、姿态、termination
和能耗。走直线 `0.8 m/s` 的横向漂移与航向偏差另与 v023 历史基线比较，但该比较
不能替代上述排序。

## 6. 遗留风险

- `f_min`、`f_max`、两个 `s_cmd` span、接触力阈值、contact scale 和两个站立
  quality scale 尚未确定；它们是模块测试与有界实验参数，不是方法自由度。
- `duty_factor=0.55` 是首个工程初值。若起步双支撑与单脚支撑时长不存在可行区，
  必须显式评审 duty 或 `STARTING` 状态，不能只调 Reward 权重。
- 二值接触阈值可能被轻触、滑擦或抖动利用，必须依靠外部步态量识别。
- SAC 混合 regime 下 Critic 必须同时观测 phase 与 command（已有，保持）。
- 当前 `heading_command=false`，训练 selector 每 4 秒重采样 command。本步 Reward 使用
  Actor 已观察到的旧 command，重采样后的 canonical command 与 phase transition 只提交
  给下一 observation/action；该两段事务已有离线 owner-boundary 测试。

## 7. 参考实现位置（调研证据）

- Unitree G1：`legged_gym/envs/g1/g1_env.py`（相位钟、`_reward_contact`、
  `_reward_feet_swing_height`）、`legged_gym/envs/base/legged_robot.py`
  （0.2 死区、`_reward_stand_still` 门控形式）。
- humanoid-gym：`humanoid/envs/custom/humanoid_env.py`（`_get_gait_phase`
  双支撑窄带 `|sin|<0.1`、`compute_ref_state` 参考关节轨迹）。
- WBC-AGILE：`agile/rl_env/mdp/commands/commands.py`（`min_vel_norm`、
  `UniformVelocityGaitBaseHeightCommand` 冻结时钟）、
  `agile/rl_env/mdp/rewards/aestetic_rewards.py`（null-cmd 门控站立组）。

## 8. 消融观察与下一步（2026-09-08）

用户对比修复前后的 checkpoint 回放，确认恢复 DR Curriculum 后，机器人在较大
速度命令下的迈步幅度更大；此前更倾向于蹭地或保守的小碎步。当前版本仍不能在
零命令下稳定站立，会缓慢失衡摔倒。以上为用户的定性回放观察，未记录量化步长。

代码检查发现：Oracle 训练中的定向执行器故障开关曾错误地与通用物理随机化课程耦合，关闭
定向故障时会把其他物理随机化课程一并关闭，导致增益、摩擦、质量、重心和关节零偏
从训练开始就使用完整扰动范围。修复后 Oracle 训练不包含定向执行器故障，其他物理扰动仍从标称
状态逐步增加。左膝增益衰减只用于 Oracle 冻结后的故障轨迹采集。用户的 checkpoint
对比支持此前解释：尚未学会稳定迈步就同时承受多种物理扰动，可能让训练收敛到
蹭地或小碎步。采用恢复 DR Curriculum 的版本作为后续基线，不再重复验证此问题。

下一步建议只重新开启原第 1 点命令死区：
`env.commands.dead_zone_enabled=true`。保留
`env.domain_rand.actuator_strength.enabled=false`、
`env.domain_rand.actuator_strength.curriculum_enabled=false` 和
`env.domain_rand.actuator_strength.group_curriculum_enabled=true`；其余 Reward、
相位、网络、训练规模和预算不变。使用独立输出目录重新训练 5000 轮，与当前
死区关闭的基线对照，观察零速/极小命令行为及较高速迈步是否保留。

死区只将极小命令归零，不直接解决精确零命令下的平衡问题。本条记录未执行
新训练，也未开启新的站立奖励或相位机制。
