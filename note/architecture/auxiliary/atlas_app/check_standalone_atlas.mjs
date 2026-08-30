import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const appDir = path.dirname(fileURLToPath(import.meta.url));
const atlasRoot = path.resolve(appDir, "../..");
const cases = [
  {
    html: "08_in_context_execution_calibration.html",
    data: "concept/08_trajectory_conditioned_execution_alignment.data.json",
    title: "08 In-Context Execution Calibration",
  },
  {
    html: "09_in_context_execution_calibration_design_inspector.html",
    data: "concept/09_in_context_execution_calibration_design_inspector.data.json",
    title: "In-Context Execution Calibration — Design Inspector",
  },
  {
    html: "10_state_conditioned_history_readout.html",
    data: "concept/10_state_conditioned_history_readout.data.json",
    title: "10 State-Conditioned History Readout",
  },
];

for (const fixture of cases) {
  const html = fs.readFileSync(path.join(atlasRoot, fixture.html), "utf8");
  if (!html.includes('data-atlas-standalone="true"')) {
    throw new Error(`${fixture.html}: missing standalone marker`);
  }
  if (html.includes("127.0.0.1")) {
    throw new Error(`${fixture.html}: still depends on a localhost server`);
  }
  if (!html.includes('id="atlas-embedded-data"')) {
    throw new Error(`${fixture.html}: missing embedded data`);
  }
  const embeddedMatch = html.match(
    /<script id="atlas-embedded-data" type="application\/json">([\s\S]*?)<\/script>/,
  );
  if (!embeddedMatch) {
    throw new Error(`${fixture.html}: embedded data element is malformed`);
  }
  const embeddedData = JSON.parse(embeddedMatch[1]);
  const sourceData = JSON.parse(fs.readFileSync(path.join(atlasRoot, fixture.data), "utf8"));
  if (JSON.stringify(embeddedData) !== JSON.stringify(sourceData)) {
    throw new Error(`${fixture.html}: embedded data is stale relative to ${fixture.data}`);
  }
  if (!html.includes(fixture.title)) {
    throw new Error(`${fixture.html}: wrong embedded page identity`);
  }
  const inlineScripts = [...html.matchAll(/<script(?![^>]*\bsrc=)([^>]*)>([\s\S]*?)<\/script>/g)]
    .filter((match) => !match[1].includes('type="application/json"'));
  for (const [, , script] of inlineScripts) {
    // Parse without executing browser-dependent code.
    new Function(script);
  }
  for (const asset of [
    "./auxiliary/atlas_app/node_modules/roughjs/bundled/rough.js",
    "./auxiliary/atlas_app/node_modules/katex/dist/katex.min.js",
    "./auxiliary/atlas_app/node_modules/katex/dist/katex.min.css",
  ]) {
    if (!html.includes(asset)) {
      throw new Error(`${fixture.html}: missing local asset ${asset}`);
    }
    const assetPath = path.resolve(atlasRoot, asset);
    if (!fs.existsSync(assetPath)) {
      throw new Error(`${fixture.html}: local asset does not exist: ${assetPath}`);
    }
  }
}

console.log("standalone Atlas entrypoints: PASS");
