import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const atlasRoot = path.resolve(__dirname, "../..");
const repoRoot = path.resolve(atlasRoot, "../..");
const vscodeCli = process.env.VSCODE_CLI_PATH
  || "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code";
const port = Number(process.env.PORT || 8767);
const clients = new Set();

const mimeTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
]);

function sendEvent(eventName) {
  for (const client of clients) {
    client.write(`event: ${eventName}\n`);
    client.write(`data: ${Date.now()}\n\n`);
  }
}

function watchDataFiles() {
  for (const dir of [
    path.join(atlasRoot, "architecture"),
    path.join(atlasRoot, "runtime"),
    path.join(atlasRoot, "concept"),
  ]) {
    if (!fs.existsSync(dir)) continue;
    fs.watch(dir, { persistent: true }, (_eventType, filename) => {
      if (filename && filename.endsWith(".data.json")) {
        sendEvent("architecture-data");
      }
    });
  }
}

function safeResolve(urlPath) {
  const cleanPath = decodeURIComponent(urlPath.split("?")[0]);
 const relativePath = cleanPath === "/" ? "index.html" : cleanPath.slice(1);
 const resolved = path.resolve(atlasRoot, relativePath);
 if (resolved !== atlasRoot && !resolved.startsWith(`${atlasRoot}${path.sep}`)) return null;
 return resolved;
}

const server = http.createServer((req, res) => {
 const requestUrl = new URL(req.url || "/", "http://127.0.0.1");
 if (requestUrl.pathname === "/healthz") {
  res.writeHead(200, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" });
  res.end(JSON.stringify({ service: "ice-cal-femr-context-atlas" }));
  return;
 }
 if (requestUrl.pathname === "/open-source") {
    const relativePath = requestUrl.searchParams.get("path") || "";
    const line = Number.parseInt(requestUrl.searchParams.get("line") || "1", 10);
    const absolutePath = path.resolve(repoRoot, relativePath);
    const insideRepo = absolutePath === repoRoot || absolutePath.startsWith(`${repoRoot}${path.sep}`);
    if (
      !insideRepo
      || !Number.isInteger(line)
      || line < 1
      || !fs.existsSync(absolutePath)
      || !fs.statSync(absolutePath).isFile()
    ) {
      res.writeHead(400, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Invalid source location");
      return;
    }
    const gotoTarget = `${absolutePath}:${line}:1`;
    console.log(`[Atlas Source Link] path=${relativePath} line=${line} dry_run=${requestUrl.searchParams.get("dry_run") === "1"}`);
    if (requestUrl.searchParams.get("dry_run") === "1") {
      res.writeHead(200, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" });
      res.end(JSON.stringify({ absolutePath, line, gotoTarget }));
      return;
    }
    // The VS Code CLI forwards --goto to an already-running editor instance.
    // macOS `open -a ... --args` may only focus that instance and drop --goto.
    if (!fs.existsSync(vscodeCli)) {
      console.error(`[Atlas Source Link] VS Code CLI not found: ${vscodeCli}`);
      res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
      res.end(`VS Code CLI not found: ${vscodeCli}`);
      return;
    }
    const opener = spawn(vscodeCli, ["--reuse-window", "--goto", gotoTarget], {
      stdio: "ignore",
    });
    opener.once("error", (error) => {
      console.error(`[Atlas Source Link] launch failed: ${error.message}`);
      if (!res.headersSent) {
        res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
        res.end(`VS Code CLI launch failed: ${error.message}`);
      }
    });
    opener.once("exit", (code) => {
      console.log(`[Atlas Source Link] VS Code CLI exit=${code}`);
      if (res.headersSent) return;
      if (code === 0) {
        res.writeHead(204, { "Cache-Control": "no-store" });
        res.end();
      } else {
        res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
        res.end(`VS Code CLI exited with code ${code}`);
      }
    });
    return;
  }
  if (req.url === "/events") {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    res.write("\n");
    clients.add(res);
    req.on("close", () => clients.delete(res));
    return;
  }

  const filePath = safeResolve(req.url || "/");
  if (!filePath) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, (error, data) => {
    if (error) {
      res.writeHead(error.code === "ENOENT" ? 404 : 500);
      res.end(error.code === "ENOENT" ? "Not found" : String(error));
      return;
    }
    res.writeHead(200, {
      "Content-Type": mimeTypes.get(path.extname(filePath)) || "application/octet-stream",
      "Cache-Control": "no-cache",
    });
    res.end(data);
  });
});

watchDataFiles();

server.listen(port, "127.0.0.1", () => {
  console.log(`ICE-Cal FEMR context atlas: http://127.0.0.1:${port}/`);
  console.log(`08 In-Context Execution Calibration: http://127.0.0.1:${port}/08_in_context_execution_calibration.html`);
  console.log(`09 Design Inspector: http://127.0.0.1:${port}/09_in_context_execution_calibration_design_inspector.html`);
  console.log(`Watching data folder: concept/`);
});
