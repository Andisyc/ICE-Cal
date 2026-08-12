import fs from "node:fs";
import path from "node:path";
import rough from "./node_modules/roughjs/bundled/rough.esm.js";

const repoRoot = path.resolve("../../../..");
const html = fs.readFileSync("architecture_atlas.html", "utf8");
const indexHtml = fs.readFileSync("../../index.html", "utf8");
const runtimeAtlas = JSON.parse(
  fs.readFileSync("../../runtime/01_unilab_runtime_atlas.data.json", "utf8"),
);
const methodToCode = JSON.parse(
  fs.readFileSync("../../architecture/02_g1_distillation_method_to_code.data.json", "utf8"),
);
const methodFigure = JSON.parse(
  fs.readFileSync("../../concept/03_g1_multiteacher_distillation_method.data.json", "utf8"),
);
const fadaMethodFigure = JSON.parse(
  fs.readFileSync("../../concept/04_fada_method_discussion.data.json", "utf8"),
);
const fadaModuleFigure = JSON.parse(
  fs.readFileSync("../../architecture/05_fada_planner_tracker_modules.data.json", "utf8"),
);
const fadaDesignDiscussion = JSON.parse(
  fs.readFileSync("../../concept/06_fada_design_detail_discussion.data.json", "utf8"),
);
const fadaDistillationFigure = JSON.parse(
  fs.readFileSync("../../concept/07_fada_planner_idm_distillation.data.json", "utf8"),
);
const contextTrackerCalibrationFigure = JSON.parse(
  fs.readFileSync("../../architecture/09_trajectory_conditioned_execution_alignment.data.json", "utf8"),
);
const contextCalibrationDesign = JSON.parse(
  fs.readFileSync("../../concept/10_fada_context_calibration_design.data.json", "utf8"),
);
const contextSearchDistillationProposal = JSON.parse(
  fs.readFileSync("../../concept/11_fada_context_search_distillation_proposal.data.json", "utf8"),
);
const contextDifferentiableTrajectory = JSON.parse(
  fs.readFileSync("../../concept/12_fada_context_differentiable_trajectory.data.json", "utf8"),
);
const contextMethodContract = fs.readFileSync(
  path.join(repoRoot, "note/fada/contracts/active/method/FADA-CONTEXT-METHOD-v003.md"),
  "utf8",
);

if (typeof rough.svg !== "function") {
  throw new Error("roughjs import succeeded but rough.svg is missing");
}
if (!html.includes('import rough from "./node_modules/roughjs/bundled/rough.esm.js";')) {
  throw new Error("architecture_atlas.html does not import local roughjs");
}
if (!html.includes('new EventSource("/events")')) {
  throw new Error("architecture_atlas.html is not wired to the auto-refresh event stream");
}
if (!html.includes('<main id="layout" class="editor-hidden">')) {
  throw new Error("architecture_atlas.html should hide the editor sidebar by default");
}
if (!html.includes('<button id="toggle-editor">Show Editor</button>')) {
  throw new Error("architecture_atlas.html default toggle label should be Show Editor");
}
if (indexHtml.includes("127.0.0.1:8766") || html.includes("127.0.0.1:8766")) {
  throw new Error("Atlas pages must preserve the port of the server that served them");
}
if (!html.includes('fetch(href, { method: "POST" })')) {
  throw new Error("reading-card source links must use the FEMR 04 POST interaction contract");
}
for (const renderer of [
  "function renderTabs",
  "function renderRepoTree",
  "function renderFlowTree",
  "function renderMethodFigure",
  "function renderInspector",
  "function renderRepositoryReadingAtlas",
]) {
  if (!html.includes(renderer)) throw new Error(`viewer missing ${renderer}`);
}
for (const requiredId of [
  'id="toggle-editor"', 'id="zoom-out"', 'id="zoom-in"',
  'id="zoom-fit"', 'id="zoom-reset"', 'id="stage"',
]) {
  if (!html.includes(requiredId)) throw new Error(`viewer missing control ${requiredId}`);
}
if (!html.includes("../../concept/03_g1_multiteacher_distillation_method.data.json")) {
  throw new Error("viewer default data path must point to the active distillation Concept Figure");
}
if (runtimeAtlas.layout !== "repository_reading_atlas") {
  throw new Error("01 UniLab Runtime Atlas must use repository_reading_atlas");
}
if (methodToCode.layout !== "repository_reading_atlas") {
  throw new Error("02 Method-to-Code Atlas must use repository_reading_atlas");
}
if (methodFigure.layout !== "method_figure") {
  throw new Error("distillation Concept Figure must use method_figure");
}
if (fadaMethodFigure.layout !== "method_figure") {
  throw new Error("FADA main-idea discussion must use method_figure");
}
if (fadaModuleFigure.layout !== "method_figure") {
  throw new Error("FADA Planner–IDM modules must reuse the method_figure visual grammar");
}
if (fadaDesignDiscussion.layout !== "design_transaction_inspector") {
  throw new Error("FADA design-detail discussion must use the Design Inspector layout");
}
if (fadaDistillationFigure.layout !== "method_figure") {
  throw new Error("FADA Planner-IDM distillation must reuse the 03 method_figure grammar");
}
if (contextTrackerCalibrationFigure.layout !== "method_figure") {
  throw new Error("09 context-conditioned Tracker calibration must use the method_figure grammar");
}
if (
  contextTrackerCalibrationFigure.status !== "superseded"
  || contextTrackerCalibrationFigure.supersededBy !== "concept/12_fada_context_differentiable_trajectory.data.json"
) {
  throw new Error("09 privileged-teacher draft must remain explicitly superseded by 12");
}
if (contextCalibrationDesign.layout !== "design_transaction_inspector") {
  throw new Error("10 Context Calibration design must use the Design Inspector layout");
}
if (
  contextSearchDistillationProposal.layout !== "design_transaction_inspector"
  || contextSearchDistillationProposal.status !== "rejected-concept"
) {
  throw new Error("11 Context search-and-distill route must remain explicitly rejected history");
}
if (
  contextDifferentiableTrajectory.layout !== "design_transaction_inspector"
  || contextDifferentiableTrajectory.status !== "accepted-method-design"
  || !String(contextDifferentiableTrajectory.authority || "").includes("FADA-CONTEXT-METHOD-v003")
) {
  throw new Error("12 differentiable-trajectory Context must map to active method v003");
}
if (!indexHtml.includes("../../concept/07_fada_planner_idm_distillation.data.json")) {
  throw new Error("Atlas index must expose the 07 FADA Planner-IDM distillation figure");
}
if (!indexHtml.includes("../../architecture/09_trajectory_conditioned_execution_alignment.data.json")) {
  throw new Error("Atlas index must expose the 09 context-conditioned Tracker calibration figure");
}
if (!indexHtml.includes("../../concept/10_fada_context_calibration_design.data.json")) {
  throw new Error("Atlas index must expose the 10 Context Calibration Design Inspector");
}
if (!indexHtml.includes("../../concept/11_fada_context_search_distillation_proposal.data.json")) {
  throw new Error("Atlas index must expose the 11 Context search-and-distill proposal");
}
if (!indexHtml.includes("../../concept/12_fada_context_differentiable_trajectory.data.json")) {
  throw new Error("Atlas index must expose the 12 differentiable-trajectory Context design");
}
if (!html.includes('layout === "design_transaction_inspector"')) {
  throw new Error("viewer does not route design_transaction_inspector data");
}
if ((fadaDesignDiscussion.cards || []).length !== 6) {
  throw new Error("FADA Design Inspector must contain all six active parent design points");
}
const commandCoverageCard = fadaDesignDiscussion.cards.find(
  (card) => card.designId === "FADA-DP-CMD-01",
);
if (!commandCoverageCard) throw new Error("FADA Design Inspector missing Command Coverage");
if (commandCoverageCard.designId !== "FADA-DP-CMD-01" || commandCoverageCard.blockId !== "CMD-COVERAGE") {
  throw new Error("FADA Command Coverage parent identity changed");
}
if ((commandCoverageCard.details || []).length !== 9) {
  throw new Error("FADA Command Coverage must expose all nine current atomic discussion details");
}
const environmentOwnership = commandCoverageCard.details.find(
  (detail) => detail.heading === "环境归属",
);
if (
  !environmentOwnership
  || !environmentOwnership.text.includes("G1StandStill")
  || !environmentOwnership.text.includes("G1WalkFlat")
) {
  throw new Error("FADA Command Coverage must preserve standing/walking environment ownership");
}
const inspectorSteps = fadaDesignDiscussion.transaction?.steps || [];
if (inspectorSteps.length !== 7 || new Set(inspectorSteps.map((step) => step.id)).size !== 7) {
  throw new Error("FADA Design Inspector must preserve the seven-step source-training spine");
}
const inspectorStepIds = new Set(inspectorSteps.map((step) => step.id));
for (const card of fadaDesignDiscussion.cards) {
  if ((card.details || []).length < 4 || (card.details || []).length > 9) {
    throw new Error(`${card.designId} must expose four to nine atomic method decisions`);
  }
  for (const stepId of card.highlightSteps || []) {
    if (!inspectorStepIds.has(stepId)) {
      throw new Error(`${card.designId} highlights missing shared step ${stepId}`);
    }
  }
}
const contextSteps = contextCalibrationDesign.transaction?.steps || [];
if (contextSteps.length !== 9 || new Set(contextSteps.map((step) => step.id)).size !== 9) {
  throw new Error("Context Calibration Design Inspector must contain nine unique transaction steps");
}
if ((contextCalibrationDesign.cards || []).length !== 8) {
  throw new Error("Context Calibration Design Inspector must contain eight parent design points");
}
const contextStepIds = new Set(contextSteps.map((step) => step.id));
for (const card of contextCalibrationDesign.cards || []) {
  if (!/^FADA-CTX-DP-0[1-8]$/.test(card.designId || "")) {
    throw new Error(`invalid Context Calibration design id ${card.designId}`);
  }
  if ((card.details || []).length < 5 || (card.details || []).length > 6) {
    throw new Error(`${card.designId} must expose five or six atomic design details`);
  }
  for (const stepId of card.highlightSteps || []) {
    if (!contextStepIds.has(stepId)) {
      throw new Error(`${card.designId} highlights missing Context step ${stepId}`);
    }
  }
  if (!card.contractAnchor || !card.implementationOwner || !card.status) {
    throw new Error(`${card.designId} missing Design Inspector governance metadata`);
  }
  for (const sourceRef of card.sourceRefs || []) {
    if (!fs.existsSync(path.join(repoRoot, sourceRef))) {
      throw new Error(`${card.designId} source does not resolve: ${sourceRef}`);
    }
  }
  for (const evidenceRef of card.evidenceRefs || []) {
    if (!fs.existsSync(path.join(repoRoot, evidenceRef))) {
      throw new Error(`${card.designId} evidence does not resolve: ${evidenceRef}`);
    }
  }
}
const proposalSteps = contextSearchDistillationProposal.transaction?.steps || [];
const proposalStepIds = new Set(proposalSteps.map((step) => step.id));
if (proposalSteps.length !== 10 || proposalStepIds.size !== 10) {
  throw new Error("Context Route 2 proposal must contain ten unique transaction steps");
}
if ((contextSearchDistillationProposal.cards || []).length !== 7) {
  throw new Error("Context Route 2 proposal must contain seven design cards");
}
for (const card of contextSearchDistillationProposal.cards || []) {
  if (!/^FADA-CTX-R2-DP-0[1-7]$/.test(card.designId || "")) {
    throw new Error(`invalid Context Route 2 proposal id ${card.designId}`);
  }
  if (card.methodContract !== "none — training route not accepted") throw new Error(`${card.designId} must not claim active-contract authority`);
  if ((card.details || []).length < 5 || (card.details || []).length > 6) {
    throw new Error(`${card.designId} must expose five or six atomic proposal details`);
  }
  for (const stepId of card.highlightSteps || []) {
    if (!proposalStepIds.has(stepId)) {
      throw new Error(`${card.designId} highlights missing Route 2 step ${stepId}`);
    }
  }
  for (const sourceRef of card.sourceRefs || []) {
    if (!fs.existsSync(path.join(repoRoot, sourceRef))) {
      throw new Error(`${card.designId} source does not resolve: ${sourceRef}`);
    }
  }
}
const differentiableSteps = contextDifferentiableTrajectory.transaction?.steps || [];
const differentiableStepIds = new Set(differentiableSteps.map((step) => step.id));
if (differentiableSteps.length !== 9 || differentiableStepIds.size !== 9) {
  throw new Error("Context differentiable-trajectory design must contain nine unique steps");
}
if ((contextDifferentiableTrajectory.cards || []).length !== 8) {
  throw new Error("Context differentiable-trajectory design must contain eight design cards");
}
for (const card of contextDifferentiableTrajectory.cards || []) {
  if (!/^FADA-CTX-DYN-DP-0[1-8]$/.test(card.designId || "")) throw new Error(`invalid differentiable Context id ${card.designId}`);
  if (card.methodContract !== "FADA-CONTEXT-METHOD-v003") throw new Error(`${card.designId} must map to Context method v003`);
  if (!contextMethodContract.includes(card.designId)) throw new Error(`${card.designId} missing from active Context method contract`);
  if ((card.details || []).length !== 4) throw new Error(`${card.designId} must expose four atomic details`);
  for (const stepId of card.highlightSteps || []) if (!differentiableStepIds.has(stepId)) throw new Error(`${card.designId} highlights missing step ${stepId}`);
  for (const sourceRef of card.sourceRefs || []) if (!fs.existsSync(path.join(repoRoot, sourceRef))) throw new Error(`${card.designId} source does not resolve: ${sourceRef}`);
  for (const evidenceRef of card.evidenceRefs || []) if (!fs.existsSync(path.join(repoRoot, evidenceRef))) throw new Error(`${card.designId} evidence does not resolve: ${evidenceRef}`);
}
const serializedContextDesign = JSON.stringify(contextCalibrationDesign);
if (/reward|奖励/i.test(serializedContextDesign)) {
  throw new Error("Context Calibration Design Inspector must not contain the rejected reward route");
}
for (const obsoleteStep of ["build-teacher", "teacher-gate"]) {
  if (serializedContextDesign.includes(obsoleteStep)) {
    throw new Error(`Context Calibration Design Inspector contains obsolete step ${obsoleteStep}`);
  }
}
if (!contextMethodContract.includes("status: active") || !contextMethodContract.includes("FADA-CTX-DYN-DP-08")) {
  throw new Error("active Context method v003 is missing status or complete design mapping");
}
const fadaModuleIds = new Set((fadaModuleFigure.nodes || []).map((node) => node.id));
if (fadaModuleIds.size !== (fadaModuleFigure.nodes || []).length) {
  throw new Error("FADA module figure contains duplicate node IDs");
}
for (const edge of fadaModuleFigure.edges || []) {
  if (!fadaModuleIds.has(edge.from) || !fadaModuleIds.has(edge.to)) {
    throw new Error(`FADA module edge references missing endpoint ${edge.from}->${edge.to}`);
  }
}

const fadaDistillationIds = new Set(
  (fadaDistillationFigure.nodes || []).map((node) => node.id),
);
if (fadaDistillationIds.size !== 7) {
  throw new Error("07 FADA distillation figure must preserve six causal nodes plus one DAgger node");
}
for (const requiredTitle of [
  "指令场景",
  "Oracle 策略",
  "训练窗口",
  "规划器 Planner",
  "逆动力学模型 IDM",
  "机器人执行",
  "异步 DAgger 聚合",
]) {
  if (!(fadaDistillationFigure.nodes || []).some((node) => node.title === requiredTitle)) {
    throw new Error(`07 FADA distillation figure missing ${requiredTitle}`);
  }
}
for (const edge of fadaDistillationFigure.edges || []) {
  if (!fadaDistillationIds.has(edge.from) || !fadaDistillationIds.has(edge.to)) {
    throw new Error(`07 FADA edge references missing endpoint ${edge.from}->${edge.to}`);
  }
}

console.log("roughjs viewer import and UniLab atlas data contracts OK");
