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

const stage1Nodes = [...nodes.values()].filter((node) => node.id.startsWith("ICA4-S1-"));
for (const node of stage1Nodes) {
  assert.equal(node.titleSize, 18, `${node.id} must use the shared Stage 1 title size`);
  assert.equal(node.summarySize, 15, `${node.id} must use the shared Stage 1 body size`);
  assert.equal(node.summaryLineHeight, 24, `${node.id} must use the shared Stage 1 line height`);
}
for (let i = 0; i < stage1Nodes.length; i += 1) {
  for (let j = i + 1; j < stage1Nodes.length; j += 1) {
    const a = stage1Nodes[i];
    const b = stage1Nodes[j];
    const overlap = a.x < b.x + b.w && a.x + a.w > b.x
      && a.y < b.y + b.h && a.y + a.h > b.y;
    assert.equal(overlap, false, `Stage 1 nodes overlap: ${a.id} and ${b.id}`);
  }
}

const stage1Edges = figure.edges.filter(
  (edge) => edge.from.startsWith("ICA4-S1-") && edge.to.startsWith("ICA4-S1-"),
);
for (const edge of stage1Edges) {
  assert.equal(edge.labelSize, 16, `${edge.from}->${edge.to} must use the shared label size`);
  assert.ok(edge.labelLines?.length >= 2, `${edge.from}->${edge.to} needs symbol and explanation lines`);
  assert.equal(edge.orthogonal, true, `${edge.from}->${edge.to} must use straight orthogonal segments`);
}
const stage1Top = Math.min(...stage1Nodes.map((node) => node.y));
const stage1Bottom = Math.max(...stage1Nodes.map((node) => node.y + node.h));
for (const edge of stage1Edges.filter((edge) => !edge.via)) {
  const widestLine = Math.max(...edge.labelLines.map((line) => [...line].reduce(
    (width, char) => width + (char.codePointAt(0) > 255 ? 16 : 8),
    0,
  )));
  const labelBox = {
    x: edge.labelX - widestLine / 2,
    y: edge.labelY - edge.labelLineHeight - edge.labelSize,
    w: widestLine,
    h: edge.labelLineHeight + edge.labelSize,
  };
  for (const node of stage1Nodes) {
    const overlap = labelBox.x < node.x + node.w && labelBox.x + labelBox.w > node.x
      && labelBox.y < node.y + node.h && labelBox.y + labelBox.h > node.y;
    assert.equal(overlap, false, `${edge.from}->${edge.to} label overlaps ${node.id}`);
  }
}
for (const edge of stage1Edges.filter((edge) => edge.via)) {
  assert.ok(
    edge.labelY < stage1Top || edge.labelY > stage1Bottom,
    `${edge.from}->${edge.to} perimeter label must clear the module band`,
  );
}

for (const prefix of ["ICA4-S2-", "ICA4-S3-"]) {
  const stageNodes = [...nodes.values()].filter((node) => node.id.startsWith(prefix));
  for (const node of stageNodes) {
    assert.equal(node.titleSize, 18, `${node.id} must use the shared title size`);
    assert.equal(node.summarySize, 15, `${node.id} must use the shared body size`);
    assert.equal(node.summaryLineHeight, 24, `${node.id} must use the shared line height`);
    assert.equal(node.shape, undefined, `${node.id} must use the shared rectangular module shape`);
  }
  for (let i = 0; i < stageNodes.length; i += 1) {
    for (let j = i + 1; j < stageNodes.length; j += 1) {
      const a = stageNodes[i];
      const b = stageNodes[j];
      const overlap = a.x < b.x + b.w && a.x + a.w > b.x
        && a.y < b.y + b.h && a.y + a.h > b.y;
      assert.equal(overlap, false, `${prefix} nodes overlap: ${a.id} and ${b.id}`);
    }
  }
  const stageEdges = figure.edges.filter(
    (edge) => edge.from.startsWith(prefix) && edge.to.startsWith(prefix),
  );
  for (const edge of stageEdges) {
    assert.equal(edge.labelSize, 16, `${edge.from}->${edge.to} must use the shared label size`);
    assert.ok(edge.labelLines?.length >= 2, `${edge.from}->${edge.to} needs symbol and explanation lines`);
    assert.equal(edge.orthogonal, true, `${edge.from}->${edge.to} must use straight orthogonal segments`);
  }
}

const stage2Context = nodes.get("ICA4-S2-CTX");
const stage2Add = nodes.get("ICA4-S2-ADD");
assert.equal(
  stage2Context.x + stage2Context.w / 2,
  stage2Add.x + stage2Add.w / 2,
  "Stage 2 Context Encoder and Latent Add must share one vertical centerline",
);
const stage2ContextEdge = figure.edges.find(
  (edge) => edge.from === "ICA4-S2-CTX" && edge.to === "ICA4-S2-ADD",
);
assert.equal(stage2ContextEdge.via, undefined, "Stage 2 coefficient path must be a single straight line");
const stage2Feedback = figure.edges.find(
  (edge) => edge.from === "ICA4-S2-LOSS" && edge.to === "ICA4-S2-CTX",
);
assert.equal(stage2Feedback.fromAnchor, "top", "Stage 2 feedback must leave Loss from the top");
assert.equal(stage2Feedback.toAnchor, "top", "Stage 2 feedback must return to Context Encoder from the top");
assert.ok(
  stage2Feedback.via.every(([, y]) => y < stage2Context.y),
  "Stage 2 feedback must stay above the Context Encoder",
);

const stage3Context = nodes.get("ICA4-S3-CTX");
const stage3Add = nodes.get("ICA4-S3-ADD");
assert.equal(
  stage3Context.x + stage3Context.w / 2,
  stage3Add.x + stage3Add.w / 2,
  "Stage 3 Context Encoder and Latent Add must share one vertical centerline",
);
const stage3ContextEdge = figure.edges.find(
  (edge) => edge.from === "ICA4-S3-CTX" && edge.to === "ICA4-S3-ADD",
);
assert.equal(stage3ContextEdge.via, undefined, "Stage 3 coefficient path must be a single straight line");

console.log("calibratable-Tracker Concept Figure: PASS");
