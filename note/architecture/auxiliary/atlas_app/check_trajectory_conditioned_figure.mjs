import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const dataPath = path.resolve(
  here,
  "../../concept/08_trajectory_conditioned_execution_alignment.data.json",
);
const figure = JSON.parse(fs.readFileSync(dataPath, "utf8"));

assert.equal(figure.layout, "method_figure");
assert.equal(figure.title, "08 In-Context Execution Calibration");
assert.ok(figure.nodes.length >= 20);
assert.ok(figure.edges.length >= 20);

const nodes = new Map(figure.nodes.map((node) => [node.id, node]));
for (const id of [
  "ICA4-S1-DATA", "ICA4-S1-ENC", "ICA4-S1-ADD", "ICA4-S1-DEC", "ICA4-S1-LOSS",
  "ICA4-S2-DATA", "ICA4-S2-CTX", "ICA4-S2-LOSS",
  "ICA4-S3-DATA", "ICA4-S3-CTX", "ICA4-S3-FIT", "ICA4-S3-PROOF",
  "ICA4-D-01", "ICA4-D-02", "ICA4-D-03", "ICA4-D-04", "ICA4-D-05",
]) {
  assert.ok(nodes.has(id), `missing current figure node ${id}`);
}

const edgeKeys = new Set(figure.edges.map((edge) => `${edge.from}->${edge.to}`));
for (const edge of figure.edges) {
  assert.ok(nodes.has(edge.from), `edge references unknown source ${edge.from}`);
  assert.ok(nodes.has(edge.to), `edge references unknown target ${edge.to}`);
}
for (const key of [
  "ICA4-S1-DATA->ICA4-S1-ENC",
  "ICA4-S2-DATA->ICA4-S2-CTX",
  "ICA4-S3-DATA->ICA4-S3-CTX",
  "ICA4-D-03->ICA4-D-04",
  "ICA4-D-04->ICA4-D-05",
]) {
  assert.ok(edgeKeys.has(key), `missing current figure edge ${key}`);
}

const text = JSON.stringify(figure);
for (const required of [
  "Stage 1", "Stage 2", "Stage 3", "Coefficient Encoder", "Direction",
  "Scale & Proof", "z +", "6 步 Action，首步为",
]) {
  assert.ok(text.includes(required), `missing current figure concept: ${required}`);
}

console.log("calibratable-Tracker Concept Figure: PASS");
