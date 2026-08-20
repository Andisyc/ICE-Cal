let activeModuleId = null;

const SVG_NS = "http://www.w3.org/2000/svg";

function basename(path) {
  return String(path || "").replace(/\/$/, "").split("/").pop();
}

function parseRouteStep(step, title) {
  const value = String(step || "");
  const prefix = value.match(/^(B\d+)\s+(.*)$/);
  const id = prefix ? prefix[1] : "B?";
  const rest = prefix ? prefix[2] : value;
  const arrow = rest.indexOf(" -> ");
  const before = arrow >= 0 ? rest.slice(0, arrow) : rest;
  const separator = before.lastIndexOf(": ");
  return {
    id,
    title: title || "正式主链步骤",
    owner: separator >= 0 ? before.slice(0, separator) : before,
  };
}

function sourceFileForRoute(module, route) {
  const owner = String(route.owner || "");
  const files = module.files || [];
  const byBasename = files.find((file) => owner.includes(basename(file.path)));
  if (byBasename) return byBasename;
  const byFunction = files.find((file) => (file.functions || []).some((fn) => {
    const symbol = String(fn).match(/([A-Za-z_]\w*)\s*\(/)?.[1] || String(fn).trim();
    return symbol && owner.includes(symbol);
  }));
  if (byFunction) return byFunction;
  return files.length === 1 ? files[0] : null;
}

function explicitRouteFunctions(owner) {
  const value = String(owner || "");
  const scoped = value.includes("::") ? value.split("::").slice(1).join("::") : value;
  return [...new Set(scoped.match(/[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\([^)]*\)/g) || [])];
}

function routeCodeDetails(module) {
  const routes = (module.mainRoute || []).map((step, index) => (
    parseRouteStep(step, (module.mainRouteTitles || [])[index])
  ));
  const usedFunctions = new Set();
  const details = routes.map((route) => {
    const file = sourceFileForRoute(module, route);
    const functions = explicitRouteFunctions(route.owner);
    functions.forEach((fn) => usedFunctions.add(fn));
    return { route, file, functions };
  });
  for (const file of module.files || []) {
    for (const fn of file.functions || []) {
      if (usedFunctions.has(fn)) continue;
      const target = details.find((detail) => detail.file === file && detail.functions.length < 3)
        || details.find((detail) => !detail.file && detail.functions.length === 0);
      if (!target) continue;
      target.file = target.file || file;
      target.functions.push(fn);
      usedFunctions.add(fn);
    }
  }
  return details;
}

function evaluationChainRowHeight(chain) {
  const rows = Math.max(1, Math.ceil((chain.functions || []).length / 5));
  return 42 + rows * 78 + Math.max(0, rows - 1) * 22 + 20;
}

function moduleDetailHeight(module) {
  if ((module.evaluationChains || []).length) {
    return 70 + module.evaluationChains.reduce((height, chain) => (
      height + evaluationChainRowHeight(chain) + 18
    ), 0) + 20;
  }
  const routeRows = Math.max(1, Math.ceil((module.mainRoute || []).length / 4));
  return 82 + routeRows * 202 + Math.max(0, routeRows - 1) * 28 + 38;
}

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

function addSourceLink(svg, addText, file, x, y, width) {
  if (!file?.sourceHref) return;
  const link = document.createElementNS(SVG_NS, "a");
  link.dataset.sourceLink = "true";
  link.setAttribute("href", file.sourceHref);
  link.addEventListener("mousedown", (event) => event.stopPropagation());
  link.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    fetch(file.sourceHref, { method: "POST" }).then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
    }).catch((error) => console.error("Failed to open source location", error));
  });
  const hit = document.createElementNS(SVG_NS, "rect");
  hit.setAttribute("x", x - 4);
  hit.setAttribute("y", y - 17);
  hit.setAttribute("width", width);
  hit.setAttribute("height", 23);
  hit.setAttribute("fill", "transparent");
  hit.setAttribute("pointer-events", "all");
  hit.style.cursor = "pointer";
  link.appendChild(hit);
  svg.appendChild(link);
  addText("↗", x + width - 12, y, { size: 13, weight: 760, fill: "#0b63b6", anchor: "end" });
}

function drawArrow(svg, points, color) {
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", points.map(([x, y], index) => `${index ? "L" : "M"}${x} ${y}`).join(" "));
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", color);
  path.setAttribute("stroke-width", "2");
  path.setAttribute("stroke-linejoin", "round");
  svg.appendChild(path);
  const [endX, endY] = points.at(-1);
  const [prevX, prevY] = points.at(-2);
  const horizontal = endY === prevY;
  const sign = horizontal ? (endX > prevX ? 1 : -1) : (endY > prevY ? 1 : -1);
  const triangle = document.createElementNS(SVG_NS, "path");
  triangle.setAttribute("d", horizontal
    ? `M${endX} ${endY} L${endX - sign * 9} ${endY - 5} L${endX - sign * 9} ${endY + 5} Z`
    : `M${endX} ${endY} L${endX - 5} ${endY - sign * 9} L${endX + 5} ${endY - sign * 9} Z`);
  triangle.setAttribute("fill", color);
  svg.appendChild(triangle);
}

function buildRegistry(data, inspector) {
  const stageByModule = new Map();
  for (const stage of inspector.stages || []) {
    for (const moduleId of stage.moduleIds || []) stageByModule.set(moduleId, stage);
  }
  const modules = [];
  for (const system of data.systems || []) {
    for (const module of system.modules || []) {
      const stage = stageByModule.get(module.id);
      modules.push({
        ...module,
        stage,
        color: stage?.color || system.color || "#64748b",
        supporting: (inspector.supportModuleIds || []).includes(module.id),
      });
    }
  }
  return { modules, moduleById: new Map(modules.map((module) => [module.id, module])) };
}

function drawModuleIndex({ svg, rc, addText, wrapText, colorWithAlpha, rerender }, modules, y) {
  const main = modules.filter((module) => !module.supporting);
  const support = modules.filter((module) => module.supporting);
  const left = 52;
  const gap = 10;
  const columns = 9;
  const width = (1856 - gap * (columns - 1)) / columns;
  const height = 58;
  addText("Module Index", left, y - 16, { size: 14, weight: 760, fill: "#4b5563" });
  main.forEach((module, index) => {
    const x = left + (index % columns) * (width + gap);
    const rowY = y + Math.floor(index / columns) * 68;
    const selected = module.id === activeModuleId;
    svg.appendChild(rc.rectangle(x, rowY, width, height, {
      stroke: module.color,
      strokeWidth: selected ? 2.7 : 1,
      roughness: selected ? 0.85 : 0.38,
      bowing: 0.24,
      fill: colorWithAlpha(module.color, selected ? 0.17 : 0.055),
      fillStyle: "solid",
    }));
    addText(module.id, x + 10, rowY + 17, { size: 9.2, weight: 730, code: true, fill: module.color });
    wrapText(module.title, x + 10, rowY + 38, width - 20, {
      size: 12.2, lineHeight: 14, charWidth: 6.2, maxLines: 2, weight: 690, fill: "#26313d",
    });
    addHitArea(svg, x, rowY, width, height, `查看 ${module.title}`, () => {
      activeModuleId = module.id;
      rerender();
    });
  });
  const supportY = y + 143;
  addText("Supporting Boundaries", left, supportY + 20, { size: 11.5, weight: 720, fill: "#6b7280" });
  support.forEach((module, index) => {
    const x = left + 170 + index * 246;
    const selected = module.id === activeModuleId;
    svg.appendChild(rc.rectangle(x, supportY, 232, 34, {
      stroke: module.color, strokeWidth: selected ? 2.2 : 0.9, roughness: 0.35,
      fill: colorWithAlpha(module.color, selected ? 0.15 : 0.05), fillStyle: "solid",
    }));
    addText(`${module.id} · ${module.title}`, x + 11, supportY + 22, {
      size: 10.4, weight: 650, code: true, fill: "#4b5563",
    });
    addHitArea(svg, x, supportY, 232, 34, `查看 ${module.title}`, () => {
      activeModuleId = module.id;
      rerender();
    });
  });
}

function drawTrainingSpine({ svg, rc, addText, wrapText, colorWithAlpha }, inspector, selectedModule, y) {
  const left = 52;
  const frameWidth = 1856;
  const cardGap = 22;
  const cardWidth = (frameWidth - 36 - cardGap * 6) / 7;
  const cardY = y + 48;
  const cardHeight = 112;
  svg.appendChild(rc.rectangle(left, y, frameWidth, 204, {
    stroke: "#8f806d", strokeWidth: 1.15, roughness: 0.55, bowing: 0.25,
    fill: "rgba(255,253,246,0.48)", fillStyle: "solid",
  }));
  addText("Training Main Loop", left + 20, y + 29, { size: 16, weight: 760, fill: "#5c5043" });
  const positions = new Map();
  inspector.stages.forEach((stage, index) => {
    const x = left + 18 + index * (cardWidth + cardGap);
    const selected = selectedModule?.stage?.id === stage.id;
    positions.set(stage.id, { x, y: cardY });
    svg.appendChild(rc.rectangle(x, cardY, cardWidth, cardHeight, {
      stroke: stage.color, strokeWidth: selected ? 2.8 : 1, roughness: selected ? 0.82 : 0.36,
      fill: colorWithAlpha(stage.color, selected ? 0.17 : 0.05), fillStyle: "solid",
    }));
    addText(String(index + 1).padStart(2, "0"), x + 12, cardY + 19, {
      size: 9.5, weight: 740, code: true, fill: stage.color,
    });
    wrapText(stage.title, x + 12, cardY + 43, cardWidth - 24, {
      size: 14, lineHeight: 17, charWidth: 7, maxLines: 2, weight: 730, fill: "#26313d",
    });
    wrapText(stage.responsibility, x + 12, cardY + 83, cardWidth - 24, {
      size: 10.2, lineHeight: 13, charWidth: 5.2, maxLines: 2, weight: 520, fill: "#5a6672",
    });
    if (index > 0) drawArrow(svg, [[x - cardGap + 4, cardY + 56], [x - 5, cardY + 56]], "#7a8793");
  });
  const update = positions.get("FC-UPDATE");
  const forward = positions.get("FC-FORWARD");
  if (update && forward) {
    drawArrow(svg, [
      [update.x + cardWidth / 2, cardY + cardHeight],
      [update.x + cardWidth / 2, y + 185],
      [forward.x + cardWidth / 2, y + 185],
      [forward.x + cardWidth / 2, cardY + cardHeight],
    ], "#b91c1c");
    addText("Actor / Critic 参数更新", (update.x + forward.x + cardWidth) / 2, y + 180, {
      size: 11, weight: 650, fill: "#b91c1c", anchor: "middle",
    });
  }
}

function drawModuleDetail(context, module, y) {
  if ((module.evaluationChains || []).length) {
    return drawEvaluationChainDetail(context, module, y);
  }
  const { svg, rc, addText, wrapText, colorWithAlpha } = context;
  const details = routeCodeDetails(module);
  const columns = 4;
  const rowCount = Math.max(1, Math.ceil(details.length / columns));
  const routeHeight = 202;
  const rowGap = 28;
  const panelHeight = 82 + rowCount * routeHeight + Math.max(0, rowCount - 1) * rowGap + 38;
  const left = 52;
  const summaryWidth = 330;
  const routeLeft = left + summaryWidth + 30;
  const routeAreaWidth = 1856 - summaryWidth - 48;
  const cardGap = 14;
  const cardWidth = (routeAreaWidth - cardGap * (columns - 1)) / columns;
  const color = module.color;
  svg.appendChild(rc.rectangle(left, y, 1856, panelHeight, {
    stroke: color, strokeWidth: 1.45, roughness: 0.5, bowing: 0.25,
    fill: colorWithAlpha(color, 0.03), fillStyle: "solid",
  }));
  addText("Selected Module", left + 24, y + 29, { size: 12, weight: 740, code: true, fill: color });
  addText(`${module.id} · ${module.title}`, left + 150, y + 29, {
    size: 17, weight: 760, fill: "#26313d",
  });
  addText(`${details.length} chain steps · ${(module.files || []).length} owner files`, left + 1832, y + 29, {
    size: 10.8, weight: 650, code: true, fill: color, anchor: "end",
  });
  const summaryY = y + 52;
  svg.appendChild(rc.rectangle(left + 18, summaryY, summaryWidth - 18, panelHeight - 72, {
    stroke: color, strokeWidth: 0.9, roughness: 0.35,
    fill: "rgba(255,254,249,0.92)", fillStyle: "solid",
  }));
  wrapText(module.owns || module.summary || "", left + 36, summaryY + 30, summaryWidth - 54, {
    size: 13, lineHeight: 17, charWidth: 6.5, maxLines: 4, weight: 540, fill: "#46515d",
  });
  addText("Owner files", left + 36, summaryY + 108, { size: 11, weight: 720, code: true, fill: color });
  (module.files || []).slice(0, 6).forEach((file, index) => {
    const fileY = summaryY + 136 + index * 23;
    addText(`${index + 1}. ${basename(file.path)}`, left + 36, fileY, {
      size: 10.8, weight: 560, code: true, fill: file.sourceHref ? "#0b63b6" : "#46515d",
    });
    addSourceLink(svg, addText, file, left + 32, fileY, summaryWidth - 58);
  });
  details.forEach(({ route, file, functions }, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const x = routeLeft + column * (cardWidth + cardGap);
    const cardY = summaryY + row * (routeHeight + rowGap);
    svg.appendChild(rc.rectangle(x, cardY, cardWidth, routeHeight, {
      stroke: color, strokeWidth: 0.9, roughness: 0.34,
      fill: "rgba(255,254,249,0.94)", fillStyle: "solid",
    }));
    addText(`${module.id} · ${module.title}`, x + 13, cardY + 21, {
      size: 9.7, weight: 730, code: true, fill: color,
    });
    addText(route.id, x + 13, cardY + 51, { size: 10.5, weight: 760, code: true, fill: color });
    wrapText(route.title, x + 46, cardY + 51, cardWidth - 59, {
      size: 13.2, lineHeight: 16, charWidth: 6.6, maxLines: 2, weight: 700, fill: "#26313d",
    });
    const fileY = cardY + 104;
    addText(file ? "file" : "owner", x + 13, fileY, { size: 9.6, weight: 720, code: true, fill: color });
    wrapText(file ? basename(file.path) : route.owner, x + (file ? 47 : 56), fileY, cardWidth - (file ? 60 : 69), {
      size: 10.4, lineHeight: 13, charWidth: 5.2, maxLines: 2, weight: 560, code: true,
      fill: file?.sourceHref ? "#0b63b6" : "#56626e",
    });
    functions.slice(0, 3).forEach((fn, fnIndex) => {
      const fnY = cardY + 142 + fnIndex * 17;
      addText("fn", x + 13, fnY, { size: 9.6, weight: 720, code: true, fill: color });
      wrapText(fn, x + 35, fnY, cardWidth - 48, {
        size: 10.4, lineHeight: 13, charWidth: 5.2, maxLines: 1, weight: 580, code: true,
        fill: file?.sourceHref ? "#0b63b6" : "#56626e",
      });
    });
    if (file) addSourceLink(svg, addText, file, x + 9, fileY, cardWidth - 18);
    if (index < details.length - 1) {
      if (column < columns - 1) {
        drawArrow(svg, [[x + cardWidth, cardY + routeHeight / 2], [x + cardWidth + cardGap - 4, cardY + routeHeight / 2]], color);
      } else {
        const nextY = cardY + routeHeight + rowGap;
        drawArrow(svg, [
          [x + cardWidth / 2, cardY + routeHeight],
          [x + cardWidth / 2, cardY + routeHeight + rowGap / 2],
          [routeLeft - 8, cardY + routeHeight + rowGap / 2],
          [routeLeft - 8, nextY + routeHeight / 2],
          [routeLeft - 2, nextY + routeHeight / 2],
        ], color);
      }
    }
  });
  return panelHeight;
}

function drawEvaluationChainDetail(context, module, y) {
  const { svg, rc, addText, wrapText, colorWithAlpha } = context;
  const left = 52;
  const width = 1856;
  const color = module.color;
  const chains = module.evaluationChains || [];
  const panelHeight = moduleDetailHeight(module);
  const labelWidth = 310;
  const routeLeft = left + labelWidth + 46;
  const routeWidth = width - labelWidth - 70;
  const columns = 5;
  const nodeGap = 14;
  const nodeWidth = (routeWidth - nodeGap * (columns - 1)) / columns;
  const nodeHeight = 78;
  const rowGap = 22;
  const statusColors = { current: "#15803d", validation: "#2563eb", legacy: "#7b8792" };

  svg.appendChild(rc.rectangle(left, y, width, panelHeight, {
    stroke: color, strokeWidth: 1.45, roughness: 0.5, bowing: 0.25,
    fill: colorWithAlpha(color, 0.03), fillStyle: "solid",
  }));
  addText("Selected Module", left + 24, y + 29, { size: 12, weight: 740, code: true, fill: color });
  addText(`${module.id} · ${module.title}`, left + 150, y + 29, {
    size: 17, weight: 760, fill: "#26313d",
  });
  addText(`${chains.length} evaluation chains · ${(module.files || []).length} owner files`, left + width - 24, y + 29, {
    size: 10.8, weight: 650, code: true, fill: color, anchor: "end",
  });

  let chainY = y + 52;
  chains.forEach((chain) => {
    const chainHeight = evaluationChainRowHeight(chain);
    const chainColor = statusColors[chain.status] || color;
    svg.appendChild(rc.rectangle(left + 18, chainY, width - 36, chainHeight, {
      stroke: chainColor, strokeWidth: 0.95, roughness: 0.34,
      fill: colorWithAlpha(chainColor, 0.025), fillStyle: "solid",
    }));
    addText(`[${chain.id}] ${chain.title}`, left + 36, chainY + 30, {
      size: 15, weight: 760, fill: "#26313d",
    });
    addText(chain.statusLabel, left + 36, chainY + 55, {
      size: 10.8, weight: 720, code: true, fill: chainColor,
    });
    wrapText(chain.purpose, left + 36, chainY + 82, labelWidth - 52, {
      size: 12.2, lineHeight: 17, charWidth: 6.2, maxLines: 3, weight: 520, fill: "#56626e",
    });

    (chain.functions || []).forEach((ref, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const x = routeLeft + column * (nodeWidth + nodeGap);
      const nodeY = chainY + 24 + row * (nodeHeight + rowGap);
      const file = (module.files || []).find((item) => item.path === ref.sourcePath);
      svg.appendChild(rc.rectangle(x, nodeY, nodeWidth, nodeHeight, {
        stroke: chainColor, strokeWidth: 0.8, roughness: 0.28,
        fill: "rgba(255,254,249,0.96)", fillStyle: "solid",
      }));
      wrapText(basename(ref.sourcePath), x + 10, nodeY + 21, nodeWidth - 20, {
        size: 9.6, lineHeight: 12, charWidth: 5, maxLines: 1, weight: 680, code: true,
        fill: file?.sourceHref ? "#0b63b6" : "#66717c",
      });
      wrapText(`${ref.name}()`, x + 10, nodeY + 46, nodeWidth - 20, {
        size: 10.6, lineHeight: 14, charWidth: 5.4, maxLines: 2, weight: 590, code: true,
        fill: file?.sourceHref ? "#0b63b6" : "#3f4b56",
      });
      if (file) addSourceLink(svg, addText, file, x + 6, nodeY + 20, nodeWidth - 12);
      if (index >= chain.functions.length - 1) return;
      if (column < columns - 1) {
        drawArrow(svg, [[x + nodeWidth, nodeY + nodeHeight / 2], [x + nodeWidth + nodeGap - 4, nodeY + nodeHeight / 2]], chainColor);
      } else {
        const nextY = nodeY + nodeHeight + rowGap;
        drawArrow(svg, [
          [x + nodeWidth / 2, nodeY + nodeHeight],
          [x + nodeWidth / 2, nodeY + nodeHeight + rowGap / 2],
          [routeLeft - 8, nodeY + nodeHeight + rowGap / 2],
          [routeLeft - 8, nextY + nodeHeight / 2],
          [routeLeft - 2, nextY + nodeHeight / 2],
        ], chainColor);
      }
    });
    chainY += chainHeight + 18;
  });
  return panelHeight;
}

export function renderModuleInspector(data, context) {
  const inspector = data.moduleInspector;
  if (!inspector?.stages?.length) throw new Error("repository atlas is missing moduleInspector.stages");
  const registry = buildRegistry(data, inspector);
  if (!activeModuleId || !registry.moduleById.has(activeModuleId)) {
    activeModuleId = inspector.defaultModuleId || data.runtimeOrder?.[0];
  }
  const selectedModule = registry.moduleById.get(activeModuleId);
  if (!selectedModule) throw new Error("moduleInspector.defaultModuleId is missing from the module registry");
  const detailHeight = moduleDetailHeight(selectedModule);
  const width = 1960;
  const detailY = 596;
  const height = detailY + detailHeight + 76;
  context.setCanvas(width, height);
  const rc = context.rough.svg(context.svg);
  const drawing = { ...context, rc };
  context.addText(inspector.title || data.title, 58, 58, { size: 30, weight: 800, fill: "#26313d" });
  context.wrapText(inspector.subtitle || "", 60, 91, width - 120, {
    size: 15, lineHeight: 20, charWidth: 7.4, maxLines: 2, weight: 520, fill: "#66717c",
  });
  drawTrainingSpine(drawing, inspector, selectedModule, 142);
  drawModuleIndex(drawing, registry.modules, 378);
  drawModuleDetail(drawing, selectedModule, detailY);
}
