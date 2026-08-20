import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const dataPath = path.resolve(
  here,
  "../../concept/09_in_context_execution_calibration_design_inspector.data.json",
);
const review = JSON.parse(fs.readFileSync(dataPath, "utf8"));

assert.equal(review.layout, "design_transaction_inspector");
assert.equal(review.cards.length, 10);
assert.deepEqual(
  review.cards.map(({ designId }) => designId),
  [
    "ICA-DP-01", "ICA-DP-02", "ICA-DP-03", "ICA-DP-04", "ICA-DP-06",
    "ICA-DP-05", "ICA-DP-07", "ICA-DP-09", "ICA-DP-10", "ICA-DP-08",
  ],
);

const stepIds = new Set(review.transaction.steps.map(({ id }) => id));
assert.equal(stepIds.size, review.transaction.steps.length);
for (const card of review.cards) {
  assert.ok(card.responsibility.length > 0, `${card.designId} needs responsibility`);
  assert.ok(card.details.length >= 3, `${card.designId} needs atomic decisions`);
  for (const stepId of card.highlightSteps) {
    assert.ok(stepIds.has(stepId), `${card.designId} references unknown step ${stepId}`);
  }
}

const text = JSON.stringify(review);
for (const required of [
  "z+Σσᵢ(cᵢ)Δzᵢ",
  "Support/Query 概念退役",
  "30 帧 State/Action",
  "2 层 Transformer",
  "每条 Δzᵢ 是 K×D（6×128）",
  "串行禁止联合训练",
  "PCHIP",
  "只执行第一步",
  "c=0 等价标称",
]) {
  assert.ok(text.includes(required), `missing current design decision: ${required}`);
}
for (const obsolete of [
  "完整 Support 与当前 Query History",
  "每个 Query 时刻独立输出一个 Δz",
  "Support 滑动窗口",
]) {
  assert.ok(!text.includes(obsolete), `superseded design remains: ${obsolete}`);
}
for (const forbidden of ["implementation-confirmed", "runtime-confirmed", "MODULE-CORRECT"]) {
  assert.ok(!text.includes(forbidden), `Design Inspector overclaims ${forbidden}`);
}

console.log("calibratable-Tracker serial three-stage Design Inspector: PASS");
