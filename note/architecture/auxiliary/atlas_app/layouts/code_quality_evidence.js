let activeStageId = null;
let activeModuleId = null;
let activeChainId = null;
let activeFunctionProjection = "chain";

const SVG_NS = "http://www.w3.org/2000/svg";
const FILE_HEADER_HEIGHT = 34;
const FUNCTION_NAME_LINE_HEIGHT = 21;
const PURPOSE_LINE_HEIGHT = 19;
const BLOCK_LINE_HEIGHT = 18;
const BLOCK_TOP_GAP = 7;
const FUNCTION_BOTTOM_GAP = 13;
const FUNCTION_COLUMN_COUNT = 2;
const FUNCTION_COLUMN_GAP = 34;
const REVIEW_FINDING_HEIGHT = 54;
const CHAIN_OVERVIEW_HEIGHT = 132;
const FUNCTION_SECTION_HEADER_HEIGHT = 42;
const FUNCTION_SECTION_GAP = 26;

function addHitArea(svg, x, y, width, height, label, onSelect) {
  const hit = document.createElementNS(SVG_NS, "rect");
  for (const [name, value] of Object.entries({ x, y, width, height })) hit.setAttribute(name, value);
  hit.setAttribute("fill", "transparent");
  hit.setAttribute("pointer-events", "all");
  hit.setAttribute("role", "button");
  hit.setAttribute("tabindex", "0");
  hit.setAttribute("aria-label", label);
  hit.style.cursor = "pointer";
  const select = (event) => {
    event.preventDefault();
    event.stopPropagation();
    onSelect();
  };
  hit.addEventListener("mousedown", (event) => event.stopPropagation());
  hit.addEventListener("click", select);
  hit.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") select(event);
  });
  svg.appendChild(hit);
}

function addSourceLink(svg, x, y, width, height, item) {
  if (!item?.sourceHref) return;
  const hit = document.createElementNS(SVG_NS, "rect");
  for (const [name, value] of Object.entries({ x, y, width, height })) hit.setAttribute(name, value);
  hit.setAttribute("fill", "transparent");
  hit.setAttribute("pointer-events", "all");
  hit.dataset.sourceLink = "true";
  if (item.id) hit.dataset.reviewId = item.id;
  hit.style.cursor = "pointer";
  hit.addEventListener("mousedown", (event) => event.stopPropagation());
  hit.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    fetch(item.sourceHref, { method: "POST" }).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
    }).catch((error) => console.error("Failed to open source location", error));
  });
  svg.appendChild(hit);
}

function drawArrow(svg, fromX, toX, y) {
  const line = document.createElementNS(SVG_NS, "path");
  line.setAttribute("d", `M${fromX} ${y} L${toX - 8} ${y}`);
  line.setAttribute("stroke", "#87919b");
  line.setAttribute("stroke-width", "2");
  line.setAttribute("fill", "none");
  svg.appendChild(line);
  const arrow = document.createElementNS(SVG_NS, "path");
  arrow.setAttribute("d", `M${toX} ${y} L${toX - 9} ${y - 5} L${toX - 9} ${y + 5} Z`);
  arrow.setAttribute("fill", "#87919b");
  svg.appendChild(arrow);
}

function fileName(sourcePath) {
  return String(sourcePath || "unknown").split("/").pop();
}

function functionStatusLabel(fn) {
  const label = {
    annotated: "已分块",
    trivial: "简单函数",
    legacy: "旧路径",
    candidate: "待标注",
  }[fn.annotationClass] || "未分类";
  return fn.annotationClass === "trivial" && fn.simpleKind ? `简单·${fn.simpleKind}` : label;
}

function functionDisplayName(fn) {
  let chainLabel = "";
  if (Array.isArray(fn.chainIds)) {
    chainLabel = fn.chainIds.length > 1
      ? `[共享 ${fn.chainIds.join("/")}] `
      : fn.chainIds.length === 1
        ? `[${fn.chainIds[0]}] `
        : "[未归链] ";
  }
  return `${chainLabel}${fn.name}() [${functionStatusLabel(fn)}]`;
}

function functionPurposeLabel(fn) {
  if (fn.annotationClass === "annotated") {
    return `功能由下方 ${fn.blocks.length} 个代码块组成`;
  }
  if (fn.annotationClass === "trivial") {
    return {
      "字段代理": "转发单一字段",
      "派生指标": "计算单一派生指标",
      "薄接口": "转发到唯一 owner",
      "纯工具": "执行小型纯变换",
    }[fn.simpleKind] || "单一步骤, 无需拆分代码块";
  }
  if (fn.annotationClass === "legacy") return "非当前正式路径, 不展开维护性标注";
  if (fn.annotationClass === "candidate") return "尚未完成白盒功能标注";
  return fn.purpose || "无函数说明";
}

function functionHeight(fn, contentWidth, measuredLineCount) {
  const nameLines = measuredLineCount(functionDisplayName(fn), contentWidth - 42, {
    charWidth: 7.2,
  });
  const purposeLines = measuredLineCount(functionPurposeLabel(fn), contentWidth - 72, {
    charWidth: 6.6,
  });
  let height = nameLines * FUNCTION_NAME_LINE_HEIGHT + purposeLines * PURPOSE_LINE_HEIGHT + 5;
  if ((fn.reviewRefs || []).length) height += 20;
  if ((fn.blocks || []).length) height += BLOCK_TOP_GAP;
  for (const block of fn.blocks || []) {
    const blockLines = measuredLineCount(block.purpose, contentWidth - 132, { charWidth: 6.4 });
    height += blockLines * BLOCK_LINE_HEIGHT + 5;
  }
  return height + FUNCTION_BOTTOM_GAP;
}

function reviewPanelHeight(module) {
  const findings = module.reviewState?.findings || [];
  return findings.length ? 46 + findings.length * REVIEW_FINDING_HEIGHT : 0;
}

function drawReviewLayer(ctx, module, x, y, width) {
  const { svg, rc, addText, wrapText, colorWithAlpha } = ctx;
  const findings = module.reviewState?.findings || [];
  if (!findings.length) return;
  const height = reviewPanelHeight(module) - 10;
  svg.appendChild(rc.rectangle(x, y, width, height, {
    stroke: "#7b8792", strokeWidth: 0.9, roughness: 0.28,
    fill: colorWithAlpha("#7b8792", 0.035), fillStyle: "solid",
  }));
  const staleCount = findings.filter((finding) => finding.currentStatus === "stale").length;
  addText("Review 标注层", x + 14, y + 23, { size: 12.5, weight: 760, fill: "#394550" });
  addText(`${findings.length - staleCount} open · ${staleCount} stale`, x + width - 14, y + 23, {
    size: 10.5, weight: 650, code: true, fill: "#66717c", anchor: "end",
  });
  findings.forEach((finding, index) => {
    const rowY = y + 47 + index * REVIEW_FINDING_HEIGHT;
    addText(`[${finding.basicType}]`, x + 14, rowY, {
      size: 11.2, weight: 760, fill: module.color,
    });
    addText(finding.id, x + 14, rowY + 21, {
      size: 9.8, weight: 650, code: true, fill: "#7a858f",
    });
    const title = finding.currentStatus === "stale" ? `${finding.title} [stale]` : finding.title;
    wrapText(title, x + 112, rowY, width - 126, {
      size: 12.1, lineHeight: 16, charWidth: 6.4, maxLines: 1, weight: 540, fill: "#4f5b66",
    });
    const targets = (finding.functionNames || []).map((name) => `${name}()`).join(" / ");
    wrapText(`${fileName(finding.sourcePath)} -> ${targets || `line ${finding.sourceLine}`}`, x + 112, rowY + 21, width - 126, {
      size: 10.8, lineHeight: 15, charWidth: 6.1, maxLines: 2, weight: 590, code: true, fill: module.color,
    });
    addSourceLink(svg, x + 8, rowY - 17, width - 16, REVIEW_FINDING_HEIGHT, finding);
  });
}

function chainOverviewHeight(module) {
  return (module.evaluationChains || []).length ? CHAIN_OVERVIEW_HEIGHT : 0;
}

function drawChainOverview(ctx, module, x, y, width) {
  const { svg, rc, addText, wrapText, colorWithAlpha, rerender } = ctx;
  const chains = module.evaluationChains || [];
  if (!chains.length) return;
  const gap = 12;
  const cardWidth = (width - gap * (chains.length - 1)) / chains.length;
  const statusColors = { current: "#15803d", validation: "#2563eb", legacy: "#7b8792" };
  addText("Evaluation 链路索引", x, y + 21, { size: 12.5, weight: 760, fill: "#394550" });
  addText("点击链路只强调对应分区, 其他函数仍保留", x + width, y + 21, {
    size: 10.5, weight: 640, code: true, fill: "#66717c", anchor: "end",
  });
  chains.forEach((chain, index) => {
    const cardX = x + index * (cardWidth + gap);
    const cardY = y + 34;
    const selected = activeChainId === chain.id;
    const chainColor = statusColors[chain.status] || module.color;
    svg.appendChild(rc.rectangle(cardX, cardY, cardWidth, 84, {
      stroke: chainColor, strokeWidth: selected ? 2.3 : 0.9, roughness: selected ? 0.65 : 0.28,
      fill: colorWithAlpha(chainColor, selected ? 0.1 : 0.025), fillStyle: "solid",
    }));
    addText(`[${chain.id}] ${chain.title}`, cardX + 11, cardY + 22, {
      size: 11.8, weight: 730, fill: "#26313d",
    });
    addText(`${chain.statusLabel} · ${chain.assignedFunctionCount} fn · ${chain.functions.length} roots`, cardX + 11, cardY + 43, {
      size: 9.8, weight: 680, code: true, fill: chainColor,
    });
    wrapText(chain.purpose, cardX + 11, cardY + 65, cardWidth - 22, {
      size: 10.6, lineHeight: 14, charWidth: 5.4, maxLines: 2, weight: 520, fill: "#56626e",
    });
    addHitArea(svg, cardX, cardY, cardWidth, 84, `高亮 ${chain.title}`, () => {
      activeFunctionProjection = "chain";
      activeChainId = selected ? null : chain.id;
      rerender();
    });
  });
}

function drawProjectionToggle(ctx, module, x, y) {
  const { svg, rc, addText, colorWithAlpha, rerender } = ctx;
  if (!(module.evaluationChains || []).length) return;
  const options = [
    { id: "chain", label: "按链路" },
    { id: "file", label: "按文件" },
  ];
  const width = 76;
  options.forEach((option, index) => {
    const selected = activeFunctionProjection === option.id;
    const optionX = x + index * (width + 8);
    svg.appendChild(rc.rectangle(optionX, y, width, 30, {
      stroke: module.color,
      strokeWidth: selected ? 2 : 0.8,
      roughness: selected ? 0.62 : 0.25,
      fill: colorWithAlpha(module.color, selected ? 0.14 : 0.025),
      fillStyle: "solid",
    }));
    addText(option.label, optionX + width / 2, y + 20, {
      size: 11.2, weight: selected ? 760 : 620, fill: selected ? module.color : "#66717c", anchor: "middle",
    });
    addHitArea(svg, optionX, y, width, 30, `切换为${option.label}`, () => {
      activeFunctionProjection = option.id;
      if (option.id === "file") activeChainId = null;
      rerender();
    });
  });
}

function treeContentHeight(functions, contentWidth, measuredLineCount) {
  let previousPath = null;
  return functions.reduce((height, fn) => {
    const fileHeader = fn.sourcePath !== previousPath ? FILE_HEADER_HEIGHT : 0;
    previousPath = fn.sourcePath;
    return height + fileHeader + functionHeight(fn, contentWidth, measuredLineCount);
  }, 0);
}

function functionColumnLayout(functions, contentWidth, measuredLineCount) {
  const columnWidth = (contentWidth - FUNCTION_COLUMN_GAP) / FUNCTION_COLUMN_COUNT;
  const targetHeight = treeContentHeight(functions, columnWidth, measuredLineCount) / FUNCTION_COLUMN_COUNT;
  const groups = Array.from({ length: FUNCTION_COLUMN_COUNT }, () => []);
  let column = 0;
  let height = 0;
  let previousPath = null;

  for (const fn of functions) {
    let itemHeight = (fn.sourcePath !== previousPath ? FILE_HEADER_HEIGHT : 0)
      + functionHeight(fn, columnWidth, measuredLineCount);
    if (column < FUNCTION_COLUMN_COUNT - 1 && groups[column].length && height + itemHeight > targetHeight) {
      column += 1;
      height = 0;
      previousPath = null;
      itemHeight = FILE_HEADER_HEIGHT + functionHeight(fn, columnWidth, measuredLineCount);
    }
    groups[column].push(fn);
    height += itemHeight;
    previousPath = fn.sourcePath;
  }

  return {
    columnWidth,
    groups,
    height: Math.max(1, ...groups.map((group) => treeContentHeight(group, columnWidth, measuredLineCount))),
  };
}

function functionProjectionSections(module) {
  const functions = module.functions || [];
  const chains = module.evaluationChains || [];
  if (!chains.length || activeFunctionProjection === "file") {
    return [{
      id: "files",
      title: "按文件完整清单",
      subtitle: "模块内全部扫描函数",
      functions,
    }];
  }

  const sections = chains.map((chain) => ({
    id: chain.id,
    title: `[${chain.id}] ${chain.title}`,
    subtitle: chain.statusLabel,
    functions: functions.filter((fn) => fn.chainIds?.length === 1 && fn.chainIds[0] === chain.id),
  }));
  sections.push({
    id: "shared",
    title: "共享函数",
    subtitle: "同时服务多条链路, 仅展示一次",
    functions: functions.filter((fn) => (fn.chainIds || []).length > 1),
  });
  sections.push({
    id: "unassigned",
    title: "未归链函数",
    subtitle: "完整保留, 不按名称猜测归属",
    functions: functions.filter((fn) => (fn.chainIds || []).length === 0),
  });
  return sections;
}

function functionProjectionLayout(module, contentWidth, measuredLineCount) {
  const sections = functionProjectionSections(module).map((section) => ({
    ...section,
    layout: functionColumnLayout(section.functions, contentWidth, measuredLineCount),
  }));
  return {
    sections,
    height: sections.reduce(
      (sum, section) => sum + FUNCTION_SECTION_HEADER_HEIGHT + section.layout.height + FUNCTION_SECTION_GAP,
      0,
    ),
  };
}

function drawFunctionColumns(ctx, module, layout, originX, originY) {
  const { svg, addText, wrapText } = ctx;
  layout.groups.forEach((group, column) => {
    const x = originX + column * (layout.columnWidth + FUNCTION_COLUMN_GAP);
    let cursorY = originY;
    let previousPath = null;
    group.forEach((fn, index) => {
      if (fn.sourcePath !== previousPath) {
        addText(fileName(fn.sourcePath), x, cursorY, {
          size: 13, weight: 780, code: true, fill: module.color,
        });
        const title = svg.lastElementChild;
        if (title) title.setAttribute("title", fn.sourcePath);
        addSourceLink(svg, x, cursorY - 19, layout.columnWidth, 25, fn);
        cursorY += FILE_HEADER_HEIGHT;
        previousPath = fn.sourcePath;
      }

      const next = group[index + 1];
      const branch = next?.sourcePath === fn.sourcePath ? "├─" : "└─";
      addText(branch, x, cursorY, { size: 14, weight: 600, code: true, fill: "#7b8792" });
      const chainMuted = activeChainId && !(fn.chainIds || []).includes(activeChainId);
      const nameLines = wrapText(functionDisplayName(fn), x + 30, cursorY, layout.columnWidth - 42, {
        size: 13.2, lineHeight: FUNCTION_NAME_LINE_HEIGHT, charWidth: 7.2,
        weight: 680, code: true, fill: chainMuted ? "#aeb5bc" : fn.sourceHref ? "#0b63b6" : "#26313d",
      });
      const purposeY = cursorY + nameLines * FUNCTION_NAME_LINE_HEIGHT + 2;
      const purposeLabel = functionPurposeLabel(fn);
      const purposeLines = wrapText(purposeLabel, x + 48, purposeY, layout.columnWidth - 72, {
        size: 12.6, lineHeight: PURPOSE_LINE_HEIGHT, charWidth: 6.6,
        weight: 520, fill: chainMuted ? "#b8bec4" : fn.annotationClass === "candidate" ? "#a16207" : "#4f5b66",
      });
      addSourceLink(
        svg,
        x + 24,
        cursorY - 18,
        layout.columnWidth - 24,
        nameLines * FUNCTION_NAME_LINE_HEIGHT + purposeLines * PURPOSE_LINE_HEIGHT + 8,
        fn,
      );
      cursorY = purposeY + purposeLines * PURPOSE_LINE_HEIGHT + 5;

      if ((fn.reviewRefs || []).length) {
        addText(`Review: ${fn.reviewRefs.join(", ")}`, x + 48, cursorY, {
          size: 10.8, weight: 650, code: true, fill: module.color,
        });
        cursorY += 20;
      }

      if ((fn.blocks || []).length) cursorY += BLOCK_TOP_GAP;
      for (const block of fn.blocks || []) {
        addText(`↳ ${block.id}`, x + 64, cursorY, {
          size: 11.7, weight: 680, code: true, fill: chainMuted ? "#aeb5bc" : module.color,
        });
        const blockLines = wrapText(block.purpose, x + 122, cursorY, layout.columnWidth - 132, {
          size: 12.1, lineHeight: BLOCK_LINE_HEIGHT, charWidth: 6.4,
          weight: 510, fill: chainMuted ? "#b8bec4" : "#59646f",
        });
        addSourceLink(
          svg,
          x + 58,
          cursorY - 17,
          layout.columnWidth - 58,
          blockLines * BLOCK_LINE_HEIGHT + 5,
          block,
        );
        cursorY += blockLines * BLOCK_LINE_HEIGHT + 5;
      }
      cursorY += FUNCTION_BOTTOM_GAP;
    });
  });
}

function drawFunctionSections(ctx, module, projection, x, y, width) {
  const { svg, rc, addText, colorWithAlpha } = ctx;
  let cursorY = y;
  projection.sections.forEach((section) => {
    const selected = activeChainId && (section.id === activeChainId || section.id === "shared");
    svg.appendChild(rc.rectangle(x, cursorY, width, 31, {
      stroke: module.color,
      strokeWidth: selected ? 1.8 : 0.65,
      roughness: selected ? 0.55 : 0.2,
      fill: colorWithAlpha(module.color, selected ? 0.09 : 0.018),
      fillStyle: "solid",
    }));
    addText(section.title, x + 11, cursorY + 21, {
      size: 12.3, weight: 760, fill: selected ? module.color : "#394550",
    });
    addText(`${section.functions.length} functions · ${section.subtitle}`, x + width - 11, cursorY + 21, {
      size: 10.4, weight: 640, code: true, fill: "#66717c", anchor: "end",
    });
    cursorY += FUNCTION_SECTION_HEADER_HEIGHT;
    drawFunctionColumns(ctx, module, section.layout, x, cursorY);
    cursorY += section.layout.height + FUNCTION_SECTION_GAP;
  });
}

function drawStages(ctx, data, y) {
  const { svg, rc, addText, wrapText, colorWithAlpha, rerender } = ctx;
  const left = 52;
  const width = 1856;
  const gap = 22;
  const cardWidth = (width - gap * 6) / 7;
  const height = 92;
  data.stages.forEach((stage, index) => {
    const x = left + index * (cardWidth + gap);
    const selected = stage.id === activeStageId;
    svg.appendChild(rc.rectangle(x, y, cardWidth, height, {
      stroke: stage.color,
      strokeWidth: selected ? 2.7 : 1,
      roughness: selected ? 0.78 : 0.34,
      fill: colorWithAlpha(stage.color, selected ? 0.16 : 0.045),
      fillStyle: "solid",
    }));
    addText(String(index + 1).padStart(2, "0"), x + 12, y + 20, {
      size: 9.5, weight: 740, code: true, fill: stage.color,
    });
    wrapText(stage.title, x + 12, y + 49, cardWidth - 24, {
      size: 13.4, lineHeight: 16, charWidth: 6.7, maxLines: 2, weight: 720, fill: "#26313d",
    });
    addHitArea(svg, x, y, cardWidth, height, `查看 ${stage.title}`, () => {
      activeStageId = stage.id;
      activeModuleId = stage.moduleIds[0];
      activeChainId = null;
      rerender();
    });
    if (index > 0) drawArrow(svg, x - gap + 4, x - 5, y + height / 2);
  });
}

function drawModuleSelector(ctx, data, y) {
  const { svg, rc, addText, wrapText, colorWithAlpha, rerender } = ctx;
  const stage = data.stages.find((item) => item.id === activeStageId);
  const modules = stage.moduleIds.map((id) => data.modules.find((module) => module.id === id));
  const left = 52;
  const gap = 12;
  const width = Math.min(280, (1856 - gap * (modules.length - 1)) / modules.length);
  addText("Modules", left, y - 14, { size: 12, weight: 740, code: true, fill: stage.color });
  modules.forEach((module, index) => {
    const x = left + index * (width + gap);
    const selected = module.id === activeModuleId;
    svg.appendChild(rc.rectangle(x, y, width, 50, {
      stroke: stage.color,
      strokeWidth: selected ? 2.3 : 0.9,
      roughness: selected ? 0.7 : 0.3,
      fill: colorWithAlpha(stage.color, selected ? 0.15 : 0.04),
      fillStyle: "solid",
    }));
    wrapText(module.title, x + 12, y + 30, width - 24, {
      size: 12.4, lineHeight: 15, charWidth: 6.2, maxLines: 1, weight: 680, fill: "#26313d",
    });
    addHitArea(svg, x, y, width, 50, `查看 ${module.title}`, () => {
      activeModuleId = module.id;
      activeChainId = null;
      rerender();
    });
  });
}

function drawFunctionTree(ctx, module, y) {
  const { svg, rc, addText, measuredLineCount, colorWithAlpha } = ctx;
  const left = 52;
  const width = 1856;
  const functions = module.functions || [];
  const hasChains = (module.evaluationChains || []).length > 0;
  const headerHeight = hasChains ? 92 : 74;
  const contentWidth = width - 48;
  const projection = functionProjectionLayout(module, contentWidth, measuredLineCount);
  const chainHeight = chainOverviewHeight(module);
  const reviewHeight = reviewPanelHeight(module);
  const panelHeight = headerHeight + chainHeight + reviewHeight + projection.height + 34;
  const blockCount = functions.reduce((sum, fn) => sum + (fn.blocks || []).length, 0);
  const classCounts = functions.reduce((counts, fn) => {
    counts[fn.annotationClass] = (counts[fn.annotationClass] || 0) + 1;
    return counts;
  }, {});
  const simpleCounts = functions.reduce((counts, fn) => {
    if (fn.annotationClass === "trivial" && fn.simpleKind) {
      counts[fn.simpleKind] = (counts[fn.simpleKind] || 0) + 1;
    }
    return counts;
  }, {});
  svg.appendChild(rc.rectangle(left, y, width, panelHeight, {
    stroke: module.color, strokeWidth: 1.35, roughness: 0.42,
    fill: colorWithAlpha(module.color, 0.025), fillStyle: "solid",
  }));
  addText(module.title, left + 24, y + 32, { size: 21, weight: 780, fill: "#26313d" });
  drawProjectionToggle(ctx, module, left + 178, y + 12);
  const fileCount = new Set(functions.map((fn) => fn.sourcePath)).size;
  addText(
    `${fileCount} files · ${functions.length} functions · ${blockCount} blocks · `
      + `${classCounts.annotated || 0} 已分块 · `
      + `简单 ${simpleCounts["字段代理"] || 0}代理/${simpleCounts["派生指标"] || 0}指标/`
      + `${simpleCounts["薄接口"] || 0}接口/${simpleCounts["纯工具"] || 0}工具 · `
      + `${classCounts.legacy || 0} 旧路径 · ${classCounts.candidate || 0} 待标注`,
    left + width - 24,
    y + 32,
    {
    size: 11, weight: 650, code: true, fill: module.color, anchor: "end",
    },
  );
  if (hasChains) {
    const exclusiveCount = functions.filter((fn) => fn.chainIds?.length === 1).length;
    const sharedCount = functions.filter((fn) => (fn.chainIds || []).length > 1).length;
    const unassignedCount = functions.filter((fn) => (fn.chainIds || []).length === 0).length;
    addText(
      `链路内 ${exclusiveCount} · 共享 ${sharedCount} · 未归链 ${unassignedCount} · 守恒 ${exclusiveCount + sharedCount + unassignedCount}/${functions.length}`,
      left + 24,
      y + 63,
      { size: 11.2, weight: 670, code: true, fill: module.color },
    );
  }
  const chainY = y + (hasChains ? 68 : 50);
  drawChainOverview(ctx, module, left + 24, chainY, width - 48);
  drawReviewLayer(ctx, module, left + 24, chainY + chainHeight, width - 48);
  drawFunctionSections(
    ctx,
    module,
    projection,
    left + 24,
    y + headerHeight + chainHeight + reviewHeight,
    contentWidth,
  );
  return panelHeight;
}

export function renderCodeQualityEvidenceAtlas(data, context) {
  if (!Array.isArray(data.stages) || !Array.isArray(data.modules)) {
    throw new Error("code quality evidence atlas requires stages and modules");
  }
  if (!activeStageId || !data.stages.some((stage) => stage.id === activeStageId)) {
    activeStageId = data.defaultStageId || data.stages[0].id;
  }
  const stage = data.stages.find((item) => item.id === activeStageId);
  if (!activeModuleId || !stage.moduleIds.includes(activeModuleId)) {
    activeModuleId = stage.moduleIds[0];
  }
  const module = data.modules.find((item) => item.id === activeModuleId);
  if (!module) throw new Error("selected code quality module is missing");
  const width = 1960;
  const treeY = 326;
  const contentWidth = 1856 - 48;
  const headerHeight = (module.evaluationChains || []).length ? 92 : 74;
  const projection = functionProjectionLayout(module, contentWidth, context.measuredLineCount);
  const height = treeY + headerHeight + chainOverviewHeight(module) + reviewPanelHeight(module) + projection.height + 110;
  context.setCanvas(width, height);
  const rc = context.rough.svg(context.svg);
  const drawing = { ...context, rc };
  context.addText(data.title, 58, 55, { size: 29, weight: 800, fill: "#26313d" });
  context.addText(data.subtitle || "", 60, 86, { size: 14.5, weight: 520, fill: "#66717c" });
  drawStages(drawing, data, 116);
  drawModuleSelector(drawing, data, 248);
  drawFunctionTree(drawing, module, treeY);
}
