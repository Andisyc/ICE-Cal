import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const dataPath = path.resolve(
  here,
  "../../concept/10_state_conditioned_history_readout.data.json",
);
const figure = JSON.parse(fs.readFileSync(dataPath, "utf8"));

assert.equal(figure.layout, "method_figure");
assert.equal(figure.title, "10 State-Conditioned History Readout");
assert.ok(figure.nodes.length >= 15);
assert.ok(figure.edges.length >= 15);

const nodes = new Map(figure.nodes.map((node) => [node.id, node]));
const conceptIds = new Set(Object.keys(figure.concepts));
for (const id of [
  "SCR-T-DATA", "SCR-T-LABEL", "SCR-T-LOSS",
  "SCR-C-PROBLEM", "SCR-C-OBS", "SCR-C-HIST", "SCR-C-ENC", "SCR-C-POOL",
  "SCR-C-POLICY", "SCR-C-ACT",
  "SCR-E-BASE", "SCR-E-DIAG", "SCR-E-SUPPORT", "SCR-E-CLAIM",
  "SCR-D-FAULT", "SCR-D-READ", "SCR-D-EXEC",
]) {
  assert.ok(nodes.has(id), `missing current figure node ${id}`);
}
for (const node of figure.nodes) {
  assert.ok(conceptIds.has(node.concept), `${node.id} references unknown concept ${node.concept}`);
}

const edgeKeys = new Set(figure.edges.map((edge) => `${edge.from}->${edge.to}`));
for (const edge of figure.edges) {
  assert.ok(nodes.has(edge.from), `edge references unknown source ${edge.from}`);
  assert.ok(nodes.has(edge.to), `edge references unknown target ${edge.to}`);
  if (edge.concept) {
    assert.ok(conceptIds.has(edge.concept), `edge references unknown concept ${edge.concept}`);
  }
}
for (const key of [
  "SCR-T-DATA->SCR-T-LABEL",
  "SCR-T-LABEL->SCR-T-LOSS",
  "SCR-T-DATA->SCR-C-HIST",
  "SCR-C-POOL->SCR-T-LOSS",
  "SCR-T-LOSS->SCR-C-POOL",
  "SCR-C-PROBLEM->SCR-C-POOL",
  "SCR-C-HIST->SCR-C-ENC",
  "SCR-C-ENC->SCR-C-POOL",
  "SCR-C-OBS->SCR-C-POOL",
  "SCR-C-POOL->SCR-C-POLICY",
  "SCR-C-POLICY->SCR-C-ACT",
  "SCR-C-POOL->SCR-E-DIAG",
  "SCR-E-BASE->SCR-E-DIAG",
  "SCR-E-DIAG->SCR-E-SUPPORT",
  "SCR-E-SUPPORT->SCR-E-CLAIM",
  "SCR-D-FAULT->SCR-D-READ",
  "SCR-D-READ->SCR-D-EXEC",
  "SCR-D-EXEC->SCR-D-READ",
]) {
  assert.ok(edgeKeys.has(key), `missing current figure edge ${key}`);
}

const text = JSON.stringify(figure);
for (const required of [
  "Query-Conditioned Readout", "query=o_t", "Static-Query Baseline",
  "Dual Diagnostics", "Support Stratification", "Privileged Frame Labels",
  "唯一新算子", "参数匹配主对照", "因果前缀", "黑盒",
]) {
  assert.ok(text.includes(required), `missing current figure concept: ${required}`);
}

// Node overlap check, applied within every horizontal band.
const allNodes = [...nodes.values()];
for (let i = 0; i < allNodes.length; i += 1) {
  for (let j = i + 1; j < allNodes.length; j += 1) {
    const a = allNodes[i];
    const b = allNodes[j];
    const overlap = a.x < b.x + b.w && a.x + a.w > b.x
      && a.y < b.y + b.h && a.y + a.h > b.y;
    assert.equal(overlap, false, `nodes overlap: ${a.id} and ${b.id}`);
  }
}

// Typography discipline: tall module nodes share the Stage-style sizes.
for (const node of allNodes.filter((entry) => entry.h >= 160)) {
  assert.equal(node.titleSize, 18, `${node.id} must use the shared title size`);
  assert.equal(node.summarySize, 15, `${node.id} must use the shared body size`);
  assert.equal(node.summaryLineHeight, 24, `${node.id} must use the shared line height`);
  assert.equal(node.shape, undefined, `${node.id} must use the shared rectangular module shape`);
}

// Centerline alignment: query source, new operator and diagnostics share one axis.
const centerX = (node) => node.x + node.w / 2;
assert.equal(centerX(nodes.get("SCR-C-OBS")), centerX(nodes.get("SCR-C-POOL")),
  "Current Observation and Query-Conditioned Readout must share one vertical centerline");
assert.equal(centerX(nodes.get("SCR-C-POOL")), centerX(nodes.get("SCR-E-DIAG")),
  "Query-Conditioned Readout and Dual Diagnostics must share one vertical centerline");
assert.equal(centerX(nodes.get("SCR-T-DATA")), centerX(nodes.get("SCR-C-HIST")),
  "Training input and History Window must share one vertical centerline");

// Connector routing: no segment may cross a non-endpoint block.
function anchor(node, side, offset = 0) {
  const cx = node.x + node.w / 2;
  const cy = node.y + node.h / 2;
  const delta = Number.isFinite(Number(offset)) ? Number(offset) : 0;
  if (side === "left") return [node.x, cy + delta];
  if (side === "right") return [node.x + node.w, cy + delta];
  if (side === "top") return [cx + delta, node.y];
  if (side === "bottom") return [cx + delta, node.y + node.h];
  return [cx, cy];
}
function edgeEndpoints(from, to, edge) {
  if (edge.fromAnchor || edge.toAnchor) {
    return [
      anchor(from, edge.fromAnchor || "right", edge.fromOffset),
      anchor(to, edge.toAnchor || "left", edge.toOffset),
    ];
  }
  const dx = centerX(to) - centerX(from);
  const dy = (to.y + to.h / 2) - (from.y + from.h / 2);
  if (Math.abs(dx) >= Math.abs(dy)) {
    return [anchor(from, dx >= 0 ? "right" : "left"), anchor(to, dx >= 0 ? "left" : "right")];
  }
  return [anchor(from, dy >= 0 ? "bottom" : "top"), anchor(to, dy >= 0 ? "top" : "bottom")];
}
function segmentCrossesRect(p, q, node) {
  // All connectors are orthogonal; test axis-aligned segment vs rect interior.
  const epsilon = 0.5;
  const x1 = node.x + epsilon;
  const x2 = node.x + node.w - epsilon;
  const y1 = node.y + epsilon;
  const y2 = node.y + node.h - epsilon;
  if (p[0] === q[0]) {
    const x = p[0];
    const lo = Math.min(p[1], q[1]);
    const hi = Math.max(p[1], q[1]);
    return x > x1 && x < x2 && hi > y1 && lo < y2;
  }
  if (p[1] === q[1]) {
    const y = p[1];
    const lo = Math.min(p[0], q[0]);
    const hi = Math.max(p[0], q[0]);
    return y > y1 && y < y2 && hi > x1 && lo < x2;
  }
  throw new Error(`non-orthogonal segment ${p} -> ${q}`);
}
for (const edge of figure.edges) {
  const from = nodes.get(edge.from);
  const to = nodes.get(edge.to);
  const [start, end] = edgeEndpoints(from, to, edge);
  const points = [start, ...(edge.via || []), end];
  for (let idx = 1; idx < points.length; idx += 1) {
    for (const node of allNodes) {
      if (node.id === edge.from || node.id === edge.to) continue;
      assert.equal(
        segmentCrossesRect(points[idx - 1], points[idx], node),
        false,
        `${edge.from}->${edge.to} segment ${idx} crosses non-endpoint block ${node.id}`,
      );
    }
  }
}

// Label placement: no label box overlaps any block.
function labelWidth(line) {
  return [...line].reduce((width, char) => width + (char.codePointAt(0) > 255 ? 16 : 8), 0);
}
for (const edge of figure.edges) {
  if (!edge.labelLines?.length || edge.labelX === undefined) continue;
  const widest = Math.max(...edge.labelLines.map(labelWidth));
  const labelBox = {
    x: edge.labelX - widest / 2,
    y: edge.labelY - edge.labelLineHeight - edge.labelSize,
    w: widest,
    h: edge.labelLineHeight + edge.labelSize,
  };
  for (const node of allNodes) {
    const overlap = labelBox.x < node.x + node.w && labelBox.x + labelBox.w > node.x
      && labelBox.y < node.y + node.h && labelBox.y + labelBox.h > node.y;
    assert.equal(overlap, false, `${edge.from}->${edge.to} label overlaps ${node.id}`);
  }
}

// Every edge label stays inside the canvas.
for (const edge of figure.edges) {
  if (edge.labelX === undefined) continue;
  assert.ok(edge.labelX > 0 && edge.labelX < figure.figureWidth,
    `${edge.from}->${edge.to} labelX outside canvas`);
  assert.ok(edge.labelY > 0 && edge.labelY < figure.figureHeight,
    `${edge.from}->${edge.to} labelY outside canvas`);
}

console.log("state-conditioned history readout Concept Figure: PASS");
