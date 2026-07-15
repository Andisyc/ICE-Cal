# Concept Figure Causal Spine Design

Status: approved layout proposal

Date: 2026-07-15

Scope: visual and routing revision of
`note/architecture/concept/05_g1_multiteacher_distillation_method.data.json`
and its `method_figure` renderer. This proposal does not change the active
distillation method contract or policy code.

## Problem

The current Concept Figure gives all design points similar visual weight and
uses feedback connectors that cross the central method path and pass through
unrelated blocks. A reader cannot immediately distinguish:

1. the main teacher-to-student distillation path;
2. the command signal that selects behavior;
3. the DAgger loop that repairs student-state distribution shift.

## Selected Composition

Use the `Causal Spine` composition selected by the human after comparing three
layouts derived from the main-figure reference library.

The primary horizontal path is:

```text
Command Intent
  -> Teacher Policies
  -> Role Data
  -> MoE Student
  -> Single Policy
  -> Robot Execution
```

`Student-State DAgger` sits below `Role Data` and `MoE Student`. The execution
feedback route travels around the bottom perimeter:

```text
Robot Execution
  -> Student-State DAgger
  -> Role Data
```

Command intent also reaches `MoE Student` through a separate upper-perimeter
route. This makes the same intent contract visible during teacher selection and
student routing without drawing through `Teacher Policies` or `Role Data`.

## Visible Semantics

Each block keeps one title and one short sentence:

| Block | Visible explanation |
| --- | --- |
| Command Intent | 无任务指令站立，任意速度指令行走 |
| Teacher Policies | 站立、行走与后续高度教师提供动作监督 |
| Role Data | 保存角色观测、intent 与教师动作 |
| MoE Student | 专家模仿对应教师，Router 学习行为选择 |
| Single Policy | 一个 checkpoint 输出当前指令所需动作 |
| Robot Execution | 闭环执行站立、行走与停止转换 |
| Student-State DAgger | 教师重标学生实际访问的状态 |

The main spine uses arrow direction and block order instead of connector labels.
Only the two non-local routes may use labels:

- upper route: `路由条件`;
- lower route: `student states` and `聚合回灌`.

## Geometry Contract

- Draw connectors before blocks so every connector stays behind block fills.
- Attach every connector to an explicit block side anchor.
- Use orthogonal segments only.
- The horizontal spine uses one shared vertical center.
- The command-routing line travels above all top-row blocks.
- The execution-feedback line travels below all blocks before entering DAgger.
- The DAgger-to-data line rises directly into the bottom of Role Data.
- Reserve at least 18 px between a connector segment and every non-endpoint
  block rectangle.
- Place labels beside or above their segment with a canvas-colored text halo.
- No connector label or block body ends with sentence punctuation.

## Programmatic Acceptance

Extend `check_distillation_atlas.mjs` with geometry validation:

1. compute every block rectangle from the same layout coordinates used by the
   renderer;
2. expand non-endpoint rectangles by the connector clearance;
3. reject any horizontal or vertical segment intersecting an expanded
   non-endpoint rectangle;
4. reject a connector without explicit source and destination anchors;
5. reject feedback routes that enter the main-spine routing corridor;
6. validate that all seven blocks and the required seven interactions remain.

Browser acceptance must confirm:

- the complete figure is readable at fit-width on desktop;
- no line crosses a non-endpoint block or visible text;
- the main distillation path is understood left to right;
- the upper command route and lower DAgger route remain visually separate.

## Non-Scope

- no active-contract semantic change;
- no policy, trainer, collector, DAgger, or playback code change;
- no checkpoint acceptance claim;
- no implementation metadata, owner paths, status, tests, or evidence on the
  Concept Figure canvas.
