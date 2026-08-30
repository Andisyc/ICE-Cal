import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const appDir = path.dirname(fileURLToPath(import.meta.url));
const atlasRoot = path.resolve(appDir, "../..");
const rendererPath = path.join(appDir, "architecture_atlas.html");
const renderer = fs.readFileSync(rendererPath, "utf8");

const pages = [
  {
    output: "08_in_context_execution_calibration.html",
    data: "concept/08_trajectory_conditioned_execution_alignment.data.json",
  },
  {
    output: "09_in_context_execution_calibration_design_inspector.html",
    data: "concept/09_in_context_execution_calibration_design_inspector.data.json",
  },
  {
    output: "10_state_conditioned_history_readout.html",
    data: "concept/10_state_conditioned_history_readout.data.json",
  },
];

function replaceRequired(source, search, replacement, label) {
  if (!source.includes(search)) {
    throw new Error(`renderer template drift: missing ${label}`);
  }
  return source.replace(search, replacement);
}

for (const page of pages) {
  const data = JSON.parse(fs.readFileSync(path.join(atlasRoot, page.data), "utf8"));
  const embedded = JSON.stringify(data).replaceAll("</script", "<\\/script");
  let html = renderer;

  html = replaceRequired(
    html,
    '<html lang="en">',
    '<html lang="en" data-atlas-standalone="true">',
    "standalone html root",
  );
  html = replaceRequired(
    html,
    "  <title>MOSAIC Architecture Atlas</title>",
    `  <title>${data.title}</title>`,
    "page title",
  );
  html = replaceRequired(
    html,
    '  <link rel="stylesheet" href="./node_modules/katex/dist/katex.min.css" />',
    '  <link rel="stylesheet" href="./auxiliary/atlas_app/node_modules/katex/dist/katex.min.css" />',
    "KaTeX stylesheet",
  );
  html = html.replace(
    /  <script>\n    \/\/ Source links require the local Atlas server;[\s\S]*?  <\/script>\n/,
    "",
  );
  html = replaceRequired(
    html,
    `  <script type="module">
    import rough from "./node_modules/roughjs/bundled/rough.esm.js";
    import katex from "./node_modules/katex/dist/katex.mjs";
    import { renderModuleInspector } from "./layouts/module_inspector.js";
    import { renderCodeQualityEvidenceAtlas } from "./layouts/code_quality_evidence.js";`,
    `  <script src="./auxiliary/atlas_app/node_modules/roughjs/bundled/rough.js"></script>
  <script src="./auxiliary/atlas_app/node_modules/katex/dist/katex.min.js"></script>
  <script id="atlas-embedded-data" type="application/json">${embedded}</script>
  <script>`,
    "module imports",
  );
  html = replaceRequired(
    html,
    `      const response = await fetch(\`${"${dataUrl}"}?t=${"${Date.now()}"}\`);
      if (!response.ok) throw new Error(\`HTTP ${"${response.status}"}\`);
      const data = await response.json();`,
    `      const embeddedElement = document.getElementById("atlas-embedded-data");
      let data;
      if (embeddedElement) {
        data = JSON.parse(embeddedElement.textContent);
      } else {
        const response = await fetch(\`${"${dataUrl}"}?t=${"${Date.now()}"}\`);
        if (!response.ok) throw new Error(\`HTTP ${"${response.status}"}\`);
        data = await response.json();
      }`,
    "data loader",
  );
  html = replaceRequired(
    html,
    '    if ("EventSource" in window) {',
    '    if (!document.getElementById("atlas-embedded-data") && "EventSource" in window) {',
    "event source guard",
  );
  html = replaceRequired(
    html,
    `    window.setInterval(() => {
      loadFromFile().catch(() => {});
    }, 1500);`,
    `    if (!document.getElementById("atlas-embedded-data")) {
      window.setInterval(() => {
        loadFromFile().catch(() => {});
      }, 1500);
    }`,
    "reload interval guard",
  );

  if (html.includes("127.0.0.1")) {
    throw new Error(`${page.output}: generated page still contains localhost dependency`);
  }
  fs.writeFileSync(path.join(atlasRoot, page.output), html);
  console.log(`built ${page.output}`);
}
