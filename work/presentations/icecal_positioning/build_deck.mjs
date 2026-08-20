import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const BUILD_DIR = "/Users/chengyuxuan/ArtiIntComVis/ICE-Cal/work/presentations/icecal_positioning";
const OUTPUT = "/Users/chengyuxuan/ArtiIntComVis/ICE-Cal/note/presentations/icecal_research_positioning_draft.pptx";
const FADA_IMAGE = "/Users/chengyuxuan/ArtiIntComVis/awesome-humanoid-execution-alignment/pptx/main_figures/2606.28476.jpg";

const W = 1280;
const H = 720;
const C = {
  ink: "#111111",
  muted: "#5F6368",
  rule: "#B8BCC4",
  panel: "#EDEDED",
  panel2: "#F6F7F8",
  blue: "#3D8DFF",
  blueLight: "#D0EDFA",
  bluePale: "#EAF5FB",
  coral: "#E66A54",
  coralPale: "#F8E2DD",
  white: "#FFFFFF",
};
const FONT = "PingFang SC";

function addText(slide, text, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: opts.name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: opts.size ?? 28,
    typeface: opts.font ?? FONT,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
    autoFit: opts.autoFit ?? "shrinkText",
  };
  return shape;
}

function addRect(slide, x, y, w, h, opts = {}) {
  return slide.shapes.add({
    geometry: opts.geometry ?? "rect",
    name: opts.name,
    position: { left: x, top: y, width: w, height: h },
    fill: opts.fill ?? C.panel,
    line: {
      style: "solid",
      fill: opts.line ?? opts.fill ?? C.panel,
      width: opts.lineWidth ?? 0,
    },
    borderRadius: opts.radius,
  });
}

function addLine(slide, x, y, w, h, opts = {}) {
  return slide.shapes.add({
    geometry: "straightConnector1",
    name: opts.name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: {
      style: "solid",
      fill: opts.color ?? C.ink,
      width: opts.width ?? 2,
      beginArrowType: opts.beginArrowType,
      endArrowType: opts.endArrowType,
    },
  });
}

function addPill(slide, text, x, y, w, fill, color = C.ink) {
  addRect(slide, x, y, w, 38, { geometry: "roundRect", fill, radius: "rounded-xl" });
  addText(slide, text, x + 12, y + 5, w - 24, 28, {
    size: 18,
    bold: true,
    color,
    align: "center",
    valign: "middle",
  });
}

function addTitle(slide, title, num, eyebrow = "ICE-CAL / RESEARCH POSITIONING") {
  addText(slide, eyebrow, 48, 32, 680, 30, { size: 16, bold: true, color: C.muted });
  addText(slide, title, 48, 74, 1150, 72, { size: 46, bold: true, autoFit: "none" });
  addText(slide, String(num).padStart(2, "0"), 1180, 650, 52, 24, {
    size: 15,
    color: C.muted,
    align: "right",
  });
}

function addNotes(slide, sources, note = "") {
  const block = sources.length
    ? `[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}\n[/Sources]`
    : "";
  slide.speakerNotes.textFrame.setText([note, block].filter(Boolean).join("\n\n"));
  slide.speakerNotes.setVisible(true);
}

async function imageBytes(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addMethodChip(slide, label, x, y, fill, color = C.ink) {
  addRect(slide, x, y, 210, 58, { geometry: "roundRect", fill, radius: "rounded-xl" });
  addText(slide, label, x + 12, y + 11, 186, 34, {
    size: 22,
    bold: true,
    color,
    align: "center",
    valign: "middle",
  });
}

function addLitDot(slide, label, x, y, fill, opts = {}) {
  addRect(slide, x, y, opts.dotSize ?? 14, opts.dotSize ?? 14, {
    geometry: "ellipse",
    fill,
    line: opts.line ?? fill,
    lineWidth: opts.lineWidth ?? 0,
  });
  addText(slide, label, x + 20, y - 6, opts.width ?? 138, 28, {
    size: opts.size ?? 16,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
  });
}

async function build() {
  await fs.mkdir(path.dirname(OUTPUT), { recursive: true });
  await fs.mkdir(path.join(BUILD_DIR, "preview"), { recursive: true });

  const deck = Presentation.create({ slideSize: { width: W, height: H } });

  // 1. Cover: Codex Grid slide-01 hierarchy, simplified.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    addText(s, "RESEARCH POSITIONING / DISCUSSION DRAFT", 48, 40, 720, 30, {
      size: 17,
      bold: true,
      color: C.muted,
    });
    addText(s, "ICE-Cal", 48, 170, 650, 100, { size: 92, bold: true, autoFit: "none" });
    addText(s, "不重新训练，只校准执行", 48, 286, 820, 88, {
      size: 48,
      bold: true,
      color: C.blue,
      autoFit: "none",
    });
    addText(s, "从一次真实执行，推断下一次执行所需的修正", 52, 424, 710, 74, {
      size: 28,
      color: C.muted,
    });
    addRect(s, 930, 104, 260, 480, { fill: C.bluePale });
    addText(s, "1", 978, 148, 160, 92, { size: 80, bold: true, color: C.blue, align: "center" });
    addText(s, "TARGET\nROLLOUT", 974, 242, 170, 72, { size: 22, bold: true, align: "center" });
    addLine(s, 1000, 348, 120, 0, { color: C.rule, width: 2 });
    addText(s, "0", 978, 374, 160, 92, { size: 80, bold: true, color: C.ink, align: "center" });
    addText(s, "OPTIMIZER\nSTEPS", 974, 468, 170, 72, { size: 22, bold: true, align: "center" });
    addNotes(s, [], "开场只说一句：ICE-Cal 不重新学习机器人要做什么，只校准它怎样把既定意图做出来。");
  }

  // 2. Scientific problem.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    addTitle(s, "真机失败，不一定是“想错了”", 2);
    addText(s, "仿真到真机的动力学变化，可能保留任务意图，却改变动作产生的真实响应。", 52, 164, 1080, 56, {
      size: 27,
      color: C.muted,
    });

    // Full command-to-motion chain. Connectors first so they stay behind nodes.
    const nodeXs = [52, 210, 368, 526, 684, 842, 1000];
    for (let i = 0; i < nodeXs.length - 1; i++) {
      addLine(s, nodeXs[i] + 126, 330, 32, 0, {
        color: i < 4 ? C.blue : C.coral,
        width: 3,
        endArrowType: "triangle",
      });
    }
    addText(s, "逆向动力学：从目标运动推断所需动作", 362, 226, 450, 32, {
      size: 20,
      bold: true,
      color: C.blue,
      align: "center",
    });
    addText(s, "正向动力学：动作经机器人产生真实运动", 756, 226, 440, 32, {
      size: 20,
      bold: true,
      color: C.coral,
      align: "center",
    });

    const labels = ["Command", "Planner", "Intent", "Tracker", "Action", "Robot", "Motion"];
    const fills = [C.panel2, C.bluePale, C.bluePale, C.ink, C.panel2, C.coralPale, C.coralPale];
    const colors = [C.ink, C.blue, C.blue, C.white, C.ink, C.coral, C.coral];
    for (let i = 0; i < nodeXs.length; i++) {
      addRect(s, nodeXs[i], 292, 126, 76, {
        fill: fills[i],
        line: i === 3 ? C.blue : fills[i],
        lineWidth: i === 3 ? 4 : 0,
      });
      addText(s, labels[i], nodeXs[i] + 5, 313, 116, 34, {
        size: i === 0 ? 17 : 21,
        bold: true,
        color: colors[i],
        align: "center",
      });
    }
    addText(s, "CALIBRATE HERE", 526, 380, 126, 26, { size: 14, bold: true, color: C.blue, align: "center" });
    addText(s, "ROBOT DEFECT", 842, 380, 126, 26, { size: 14, bold: true, color: C.coral, align: "center" });

    addRect(s, 246, 438, 330, 66, { fill: C.bluePale });
    addText(s, "理想：Intent ≈ Motion", 268, 454, 286, 34, { size: 23, bold: true, color: C.blue, align: "center" });
    addRect(s, 704, 438, 330, 66, { fill: C.coralPale });
    addText(s, "缺陷：Intent ≠ Motion", 726, 454, 286, 34, { size: 23, bold: true, color: C.coral, align: "center" });

    addRect(s, 52, 548, 1136, 82, { fill: C.ink });
    addText(s, "缺陷改变 Action → Robot → Motion；冻结 Planner，校准 Tracker。", 84, 568, 1072, 42, {
      size: 29,
      bold: true,
      color: C.white,
      align: "center",
    });
    addNotes(s, ["https://arxiv.org/html/2606.28476"], "Planner 负责产生任务 Intent；Tracker 承担 Intent 到 Action 的逆向映射；Robot 是 Action 到 Motion 的正向动力学。缺陷改变正向动力学，因此在保持 Intent 的前提下校准 Tracker。");
  }

  // 3. Literature map: adaptation mechanism x calibration location.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    addTitle(s, "现有工作如何适应部署偏差", 3);
    addText(s, "不同方法既改变适应方式，也选择不同的校准位置。", 52, 148, 920, 38, {
      size: 24,
      color: C.muted,
    });

    const ox = 224;
    const oy = 586;
    const axW = 914;
    const rowYs = [280, 410, 530];
    addLine(s, ox, oy, axW, 0, { color: C.ink, width: 2, endArrowType: "triangle" });
    for (const y of rowYs) addLine(s, ox, y, axW, 0, { color: C.rule, width: 1 });
    addLine(s, 502, 244, 0, 342, { color: C.rule, width: 1 });
    addLine(s, 826, 244, 0, 342, { color: C.rule, width: 1 });

    addText(s, "Motion Prior / Planner", 44, 266, 158, 34, { size: 18, bold: true, align: "right" });
    addText(s, "Tracker / Policy", 44, 396, 158, 34, { size: 18, bold: true, align: "right" });
    addText(s, "Execution Mapping", 44, 516, 158, 34, { size: 18, bold: true, align: "right" });
    addText(s, "源侧鲁棒化", 248, 604, 180, 30, { size: 19, bold: true, align: "center" });
    addText(s, "在线反馈 / 推理适应", 548, 604, 230, 30, { size: 19, bold: true, align: "center" });
    addText(s, "目标域优化", 920, 604, 180, 30, { size: 19, bold: true, align: "center" });

    addLitDot(s, "PolySim '25", 306, 402, C.rule, { width: 120 });
    addLitDot(s, "ReactiveBFM '26", 548, 272, C.blue, { width: 150 });
    addLitDot(s, "RLPF '25", 878, 272, C.coral, { width: 108 });
    addLitDot(s, "REFINE-DP '26", 1002, 304, C.coral, { width: 140 });
    addLitDot(s, "RMA '21", 532, 392, C.blue, { width: 90 });
    addLitDot(s, "SplitAdapter '26", 638, 424, C.blue, { width: 140 });
    addLitDot(s, "RGB '26", 748, 382, C.blue, { width: 90 });
    addLitDot(s, "HERO '26", 728, 456, C.blue, { width: 100 });
    addLitDot(s, "MG + Tracker '26", 884, 392, C.coral, { width: 150 });
    addLitDot(s, "HALO '26", 1030, 440, C.coral, { width: 100 });
    addLitDot(s, "AnyBody '26", 572, 522, C.blue, { width: 120 });
    addLitDot(s, "FADA '26", 970, 522, C.coral, { width: 100, bold: true });
    addRect(s, 692, 500, 164, 58, { geometry: "roundRect", fill: C.ink, line: C.blue, lineWidth: 4, radius: "rounded-xl" });
    addText(s, "ICE-Cal", 712, 510, 124, 34, { size: 24, bold: true, color: C.white, align: "center" });

    addRect(s, 52, 654, 14, 14, { geometry: "ellipse", fill: C.rule });
    addText(s, "训练时增强", 74, 647, 126, 28, { size: 15, color: C.muted });
    addRect(s, 220, 654, 14, 14, { geometry: "ellipse", fill: C.blue });
    addText(s, "部署时推理", 242, 647, 126, 28, { size: 15, color: C.muted });
    addRect(s, 388, 654, 14, 14, { geometry: "ellipse", fill: C.coral });
    addText(s, "目标域优化", 410, 647, 126, 28, { size: 15, color: C.muted });
    addText(s, "12 篇代表工作；坐标表示主要校准机制，不表示论文只有这一项贡献。", 620, 646, 568, 30, {
      size: 15,
      color: C.muted,
      align: "right",
    });
    addNotes(s, [
      "https://arxiv.org/abs/2510.01708",
      "https://arxiv.org/abs/2606.30362",
      "https://arxiv.org/abs/2506.12769",
      "https://arxiv.org/abs/2603.13707",
      "https://arxiv.org/abs/2107.04034",
      "https://arxiv.org/abs/2606.03297",
      "https://arxiv.org/abs/2606.25123",
      "https://arxiv.org/abs/2602.16705",
      "https://arxiv.org/abs/2604.17335",
      "https://arxiv.org/abs/2603.15084",
      "https://arxiv.org/abs/2606.29209",
      "https://arxiv.org/abs/2606.28476",
    ], "先展示文献覆盖面，再讲 ICE-Cal。横轴不是简单的计算量，而是目标域信息进入策略的方式；纵轴是主要校准位置。");
  }

  // 4. Two independent literature lines converging on the ICE-Cal question.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    addTitle(s, "两条文献线，汇合成一个研究问题", 4);
    addText(s, "先分别讲清“怎么校准”和“在哪里校准”，最后只保留一个交叉点。", 52, 148, 1020, 38, {
      size: 24,
      color: C.muted,
    });

    addPill(s, "A  校准方式", 52, 210, 170, C.ink, C.white);
    const cols = [252, 562, 872];
    const modeTitles = ["源侧一次训练", "部署时推理", "目标域再优化"];
    const modeWorks = ["DR / PolySim", "RMA · SplitAdapter\nReactiveBFM · RGB", "FADA · RLPF\nREFINE-DP · HALO"];
    for (let i = 0; i < 3; i++) {
      addRect(s, cols[i], 202, 268, 146, { fill: i === 1 ? C.bluePale : C.panel2 });
      addText(s, modeTitles[i], cols[i] + 18, 220, 232, 34, { size: 22, bold: true, color: i === 1 ? C.blue : C.ink, align: "center" });
      addText(s, modeWorks[i], cols[i] + 18, 268, 232, 58, { size: 17, color: C.muted, align: "center" });
      if (i < 2) addLine(s, cols[i] + 274, 274, 26, 0, { color: C.rule, width: 2, endArrowType: "triangle" });
    }

    addPill(s, "B  校准位置", 52, 406, 170, C.ink, C.white);
    const posTitles = ["Planner / Motion Prior", "Tracker / Policy", "Execution Mapping"];
    const posWorks = ["ReactiveBFM · RLPF\nREFINE-DP", "RMA · SplitAdapter · RGB\nMG+Tracker · HERO", "FADA · AnyBody\nICE-Cal"];
    for (let i = 0; i < 3; i++) {
      addRect(s, cols[i], 398, 268, 146, { fill: i === 2 ? C.bluePale : C.panel2, line: i === 2 ? C.blue : undefined, lineWidth: i === 2 ? 3 : 0 });
      addText(s, posTitles[i], cols[i] + 16, 416, 236, 34, { size: 20, bold: true, color: i === 2 ? C.blue : C.ink, align: "center" });
      addText(s, posWorks[i], cols[i] + 16, 464, 236, 58, { size: 17, color: C.muted, align: "center" });
      if (i < 2) addLine(s, cols[i] + 274, 470, 26, 0, { color: C.rule, width: 2, endArrowType: "triangle" });
    }

    addRect(s, 52, 588, 1136, 66, { fill: C.blue });
    addText(s, "ICE-Cal 的候选贡献 = 目标域轨迹证据 × 推理时适应 × 执行层受限校准", 78, 604, 1084, 36, {
      size: 27,
      bold: true,
      color: C.white,
      align: "center",
    });
    addNotes(s, [
      "/Users/chengyuxuan/ArtiIntComVis/awesome-humanoid-execution-alignment/papers.csv",
      "/Users/chengyuxuan/ArtiIntComVis/awesome-humanoid-execution-alignment/research_chunks/05_failed_rollout_in_context_alignment_claim_matrix.md",
    ], "这页口头只讲两个问题：第一，目标域信息怎样进入方法；第二，它最终改动策略的哪个位置。不要逐篇介绍。");
  }

  // 5. ICE-Cal method flow.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    addTitle(s, "一次真实执行，校准下一次执行", 5);

    // Connectors first, then nodes.
    addLine(s, 282, 242, 58, 0, { color: C.blue, width: 3, endArrowType: "triangle" });
    addLine(s, 440, 242, 60, 0, { color: C.blue, width: 3, endArrowType: "triangle" });
    addLine(s, 650, 242, 40, 0, { color: C.blue, width: 3, endArrowType: "triangle" });
    addLine(s, 262, 430, 18, 0, { color: C.ink, width: 2, endArrowType: "triangle" });
    addLine(s, 400, 430, 30, 0, { color: C.ink, width: 2, endArrowType: "triangle" });
    addLine(s, 520, 430, 20, 0, { color: C.ink, width: 2, endArrowType: "triangle" });
    addLine(s, 690, 430, 40, 0, { color: C.ink, width: 2, endArrowType: "triangle" });
    addLine(s, 800, 430, 18, 0, { color: C.ink, width: 2, endArrowType: "triangle" });
    addLine(s, 872, 430, 48, 0, { color: C.ink, width: 2, endArrowType: "triangle" });
    addLine(s, 1050, 430, 40, 0, { color: C.ink, width: 2, endArrowType: "triangle" });
    addLine(s, 352, 530, 38, 0, { color: C.blue, width: 2, endArrowType: "triangle" });
    addLine(s, 520, 530, 40, 0, { color: C.blue, width: 2, endArrowType: "triangle" });
    addLine(s, 640, 530, 205, 0, { color: C.blue, width: 2 });
    addLine(s, 845, 430, 0, 100, { color: C.blue, width: 2, beginArrowType: "triangle" });

    addText(s, "第一次 Rollout", 52, 160, 280, 38, { size: 25, bold: true, color: C.blue });
    addRect(s, 52, 208, 230, 68, { fill: C.panel2 });
    addText(s, "冻结策略 / 未校准执行", 64, 225, 206, 34, { size: 20, bold: true, align: "center" });
    addRect(s, 340, 208, 100, 68, { fill: C.bluePale });
    addText(s, "记录", 352, 225, 76, 34, { size: 23, bold: true, color: C.blue, align: "center" });
    addRect(s, 500, 194, 150, 96, { fill: C.blue });
    addText(s, "完整 Support", 510, 223, 130, 38, { size: 19, bold: true, color: C.white, align: "center" });
    addRect(s, 690, 208, 498, 68, { fill: C.panel2 });
    addText(s, "State + Executed Action + Planner Intent", 708, 225, 462, 34, { size: 20, bold: true, align: "center" });

    addText(s, "第二次 Rollout / 每个控制周期", 52, 332, 460, 38, { size: 25, bold: true });
    addRect(s, 52, 395, 210, 70, { fill: C.panel2 });
    addText(s, "State History\n+ Command", 66, 407, 182, 48, { size: 19, bold: true, align: "center" });
    addRect(s, 280, 395, 120, 70, { fill: C.bluePale });
    addText(s, "Frozen\nPlanner", 298, 406, 84, 48, { size: 19, bold: true, color: C.blue, align: "center" });
    addRect(s, 430, 395, 90, 70, { fill: C.bluePale });
    addText(s, "Intent", 442, 414, 66, 32, { size: 20, bold: true, color: C.blue, align: "center" });
    addRect(s, 540, 395, 150, 70, { fill: C.panel2 });
    addText(s, "Tracker Encoder\n+ current histories", 552, 406, 126, 48, { size: 16, bold: true, align: "center" });
    addRect(s, 730, 395, 70, 70, { fill: C.panel2 });
    addText(s, "zₜ", 742, 414, 46, 32, { size: 22, bold: true, align: "center" });
    addRect(s, 818, 403, 54, 54, { geometry: "ellipse", fill: C.blue });
    addText(s, "+", 829, 408, 32, 36, { size: 28, bold: true, color: C.white, align: "center" });
    addRect(s, 920, 395, 130, 70, { fill: C.ink });
    addText(s, "Frozen\nDecoder", 938, 406, 94, 48, { size: 19, bold: true, color: C.white, align: "center" });
    addRect(s, 1090, 395, 98, 70, { fill: C.blue });
    addText(s, "Action[0]", 1102, 414, 74, 32, { size: 18, bold: true, color: C.white, align: "center" });

    addRect(s, 52, 495, 300, 70, { fill: C.panel2 });
    addText(s, "完整 Support + 当前\nState / Action History", 68, 506, 268, 48, { size: 18, bold: true, align: "center" });
    addRect(s, 390, 495, 130, 70, { fill: C.bluePale });
    addText(s, "Context\nEncoder", 408, 506, 94, 48, { size: 19, bold: true, color: C.blue, align: "center" });
    addRect(s, 560, 495, 80, 70, { fill: C.blue });
    addText(s, "Δzₜ", 570, 514, 60, 32, { size: 20, bold: true, color: C.white, align: "center" });

    addRect(s, 52, 592, 1136, 52, { fill: C.bluePale });
    addText(s, "Planner 决定 Intent；Context 只修正执行 latent；Decoder 仍是唯一 Action 生成器。", 74, 603, 1092, 30, {
      size: 24,
      bold: true,
      color: C.blue,
      align: "center",
    });
    addNotes(s, [
      "/Users/chengyuxuan/ArtiIntComVis/ICE-Cal/note/fada/contracts/active/method/FADA-CONTEXT-METHOD-v006.md",
      "/Users/chengyuxuan/ArtiIntComVis/ICE-Cal/note/architecture/concept/09_in_context_execution_calibration_design_inspector.data.json",
    ]);
  }

  // 6. Scientific concept.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    addTitle(s, "科研概念：执行受限的摊销式适应", 6);
    addText(s, "Execution-Constrained Amortized Adaptation", 52, 148, 900, 44, {
      size: 26,
      bold: true,
      color: C.blue,
    });

    addRect(s, 52, 232, 548, 290, { fill: C.panel2 });
    addText(s, "01", 78, 258, 70, 42, { size: 30, bold: true, color: C.blue });
    addText(s, "摊销适应", 150, 256, 300, 44, { size: 34, bold: true });
    addText(s, "把每个目标域都要执行的优化，\n提前学成 Context 的一次前向推理。", 78, 330, 460, 88, {
      size: 25,
      color: C.muted,
    });
    addText(s, "Target rollout  →  Δzₜ", 78, 450, 460, 36, { size: 24, bold: true, color: C.blue });

    addRect(s, 628, 232, 560, 290, { fill: C.panel2 });
    addText(s, "02", 654, 258, 70, 42, { size: 30, bold: true, color: C.blue });
    addText(s, "校准权限受限", 728, 256, 380, 44, { size: 34, bold: true });
    addText(s, "Context 不接管 Planner，\n只调整 Intent 到 Action 的实现方式。", 654, 330, 470, 88, {
      size: 25,
      color: C.muted,
    });
    addText(s, "Intent authority stays frozen", 654, 450, 460, 36, { size: 24, bold: true, color: C.blue });

    addRect(s, 52, 558, 1136, 78, { fill: C.ink });
    addText(s, "更低部署成本  +  更明确的行为权限", 84, 577, 1072, 42, {
      size: 32,
      bold: true,
      color: C.white,
      align: "center",
    });
    addNotes(s, [
      "https://arxiv.org/html/2606.28476",
      "https://ashish-kmr.github.io/rma-legged-robots/",
      "https://arxiv.org/html/2606.03297",
    ], "本页术语是基于现有方法差异提出的候选概念，不宣称已被实验验证。");
  }

  // 7. Falsifiable evidence plan.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    addTitle(s, "贡献必须由三组对照共同证明", 7);
    addText(s, "相同目标域数据预算，相同源策略，只改变适应方式与校准位置。", 52, 150, 1000, 42, {
      size: 25,
      color: C.muted,
    });

    const xs = [52, 436, 820];
    const names = ["FADA", "RMA / Split-style", "ICE-Cal"];
    const fills = [C.panel2, C.panel2, C.bluePale];
    const lines = [
      ["目标域微调", "执行模块", "精度上界 / 优化成本"],
      ["无需更新", "整体 Policy Context", "行为漂移风险"],
      ["无需更新", "执行 latent", "效率 + 权限约束"],
    ];
    for (let i = 0; i < 3; i++) {
      addRect(s, xs[i], 230, 340, 272, { fill: fills[i], line: i === 2 ? C.blue : fills[i], lineWidth: i === 2 ? 3 : 0 });
      addText(s, names[i], xs[i] + 24, 254, 292, 42, {
        size: 30,
        bold: true,
        color: i === 2 ? C.blue : C.ink,
        align: "center",
      });
      addLine(s, xs[i] + 34, 318, 272, 0, { color: C.rule, width: 1 });
      addText(s, lines[i][0], xs[i] + 28, 342, 284, 34, { size: 22, bold: true, align: "center" });
      addText(s, lines[i][1], xs[i] + 28, 390, 284, 34, { size: 22, align: "center", color: C.muted });
      addText(s, lines[i][2], xs[i] + 28, 444, 284, 36, { size: 19, align: "center", color: i === 2 ? C.blue : C.coral, bold: true });
    }
    addText(s, "必须同时回答", 52, 552, 220, 38, { size: 24, bold: true });
    addText(s, "1  精度是否接近 FADA？", 286, 552, 260, 38, { size: 19, bold: true });
    addText(s, "2  是否减少无关行为漂移？", 566, 552, 300, 38, { size: 19, bold: true });
    addText(s, "3  是否降低适应时间与状态成本？", 884, 552, 304, 38, { size: 19, bold: true });
    addText(s, "若只证明“能运行”，而不能回答以上问题，ICE-Cal 的贡献不成立。", 52, 622, 1000, 32, {
      size: 19,
      color: C.coral,
      bold: true,
    });
    addNotes(s, [
      "https://arxiv.org/html/2606.28476",
      "https://ashish-kmr.github.io/rma-legged-robots/",
      "https://arxiv.org/html/2606.03297",
    ]);
  }

  // 8. Close.
  {
    const s = deck.slides.add();
    s.background.fill = C.ink;
    addText(s, "ICE-CAL / CLAIM", 52, 42, 480, 32, { size: 17, bold: true, color: C.blueLight });
    addText(s, "不是重新训练机器人。", 52, 150, 1060, 68, { size: 52, bold: true, color: C.white, autoFit: "none" });
    addText(s, "不是让 Context 接管策略。", 52, 244, 1060, 68, { size: 52, bold: true, color: C.white, autoFit: "none" });
    addText(s, "只修正如何把既定意图做出来。", 52, 338, 1136, 80, { size: 52, bold: true, color: C.blueLight, autoFit: "none" });
    addLine(s, 52, 474, 1136, 0, { color: "#3A3A3A", width: 2 });
    addText(s, "One target rollout  /  Zero optimizer steps  /  Execution layer only", 52, 520, 1136, 48, {
      size: 27,
      color: C.white,
    });
    addText(s, "候选贡献已清晰；新颖性与有效性仍需对照实验完成。", 52, 616, 900, 34, {
      size: 20,
      color: "#BFC3C7",
    });
    addNotes(s, [], "结尾不要说 ICE-Cal 已经优于现有工作，只说我们已经找到可证伪、可比较的贡献对象。");
  }

  for (const [index, slide] of deck.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await deck.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(BUILD_DIR, "preview", `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(BUILD_DIR, "preview", `${stem}.layout.json`), await layout.text());
  }

  const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(BUILD_DIR, "preview", "montage.webp"), new Uint8Array(await montage.arrayBuffer()));
  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(OUTPUT);
  console.log(`Wrote ${OUTPUT}`);
}

build().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
