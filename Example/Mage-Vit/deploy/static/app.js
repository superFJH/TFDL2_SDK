const form = document.querySelector("#inference-form");
const videoInput = document.querySelector("#video-input");
const dropZone = document.querySelector("#drop-zone");
const dropCopy = document.querySelector("#drop-copy");
const videoPreview = document.querySelector("#video-preview");
const fileMeta = document.querySelector("#file-meta");
const runButton = document.querySelector("#run-button");
const cancelButton = document.querySelector("#cancel-button");
const answer = document.querySelector("#answer");
const runStatus = document.querySelector("#run-status");
const tokenCount = document.querySelector("#token-count");
const elapsed = document.querySelector("#elapsed");
const metrics = {
  ttft: document.querySelector("#metric-ttft"),
  decode: document.querySelector("#metric-decode"),
  tps: document.querySelector("#metric-tps"),
  total: document.querySelector("#metric-total"),
};

let controller = null;
let timer = null;
let objectUrl = null;
let startedAt = 0;

const formatSeconds = (value) =>
  value == null || !Number.isFinite(Number(value))
    ? "—"
    : `${Number(value).toFixed(2)} s`;

const formatBytes = (bytes) => {
  if (!bytes) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), 3);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
};

function setVideo(file) {
  if (!file) return;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  videoPreview.src = objectUrl;
  dropZone.classList.add("has-video");
  fileMeta.textContent = `${file.name} · ${formatBytes(file.size)}`;
}

videoInput.addEventListener("change", () => setVideo(videoInput.files[0]));
["dragenter", "dragover"].forEach((name) =>
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  }),
);
["dragleave", "drop"].forEach((name) =>
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  }),
);
dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  videoInput.files = transfer.files;
  setVideo(file);
});

function resetRun() {
  answer.className = "answer streaming";
  answer.textContent = "";
  tokenCount.textContent = "0 tokens";
  Object.values(metrics).forEach((node) => { node.textContent = "—"; });
  document.querySelectorAll("#stages > div").forEach((node) => {
    node.className = "";
    node.querySelector("time").textContent = "—";
  });
  runStatus.className = "status-pill running";
  runStatus.textContent = "准备任务";
  form.classList.add("running");
  runButton.disabled = true;
  cancelButton.disabled = false;
  startedAt = performance.now();
  clearInterval(timer);
  timer = setInterval(() => {
    elapsed.textContent = `${((performance.now() - startedAt) / 1000).toFixed(2)} s`;
  }, 80);
}

function finishRun(status, isError = false) {
  clearInterval(timer);
  timer = null;
  form.classList.remove("running");
  runButton.disabled = false;
  cancelButton.disabled = true;
  answer.classList.remove("streaming");
  runStatus.className = `status-pill${isError ? " error" : ""}`;
  runStatus.textContent = status;
  controller = null;
}

function stageEvent(event) {
  const node = document.querySelector(`[data-stage="${event.stage}"]`);
  if (!node) return;
  if (event.type === "stage_start") {
    node.classList.add("active");
    runStatus.textContent = node.querySelector("span").textContent;
  } else {
    node.classList.remove("active");
    node.classList.add("done");
    node.querySelector("time").textContent = formatSeconds(event.seconds);
  }
}

function consumeEvent(event) {
  if (event.type === "stage_start" || event.type === "stage_done") {
    stageEvent(event);
    return;
  }
  if (event.type === "token") {
    answer.textContent = event.replace ? event.text : `${answer.textContent}${event.text_delta}`;
    tokenCount.textContent = `${event.token_index + 1} tokens`;
    if (event.time_to_first_token_seconds != null) {
      metrics.ttft.textContent = formatSeconds(event.time_to_first_token_seconds);
    }
    answer.scrollTop = answer.scrollHeight;
    return;
  }
  if (event.type === "decoder_done") {
    metrics.tps.textContent = `${Number(event.ort_tokens_per_second).toFixed(2)}`;
    return;
  }
  if (event.type === "done") {
    answer.textContent = event.text;
    tokenCount.textContent = `${event.generated_tokens} tokens`;
    metrics.ttft.textContent = formatSeconds(event.time_to_first_token_seconds);
    metrics.decode.textContent = formatSeconds(event.decode_seconds);
    metrics.tps.textContent = `${Number(event.ort_tokens_per_second).toFixed(2)}`;
    metrics.total.textContent = formatSeconds(event.total_seconds);
    elapsed.textContent = formatSeconds(event.total_seconds);
    finishRun("完成");
    return;
  }
  if (event.type === "error") {
    answer.className = "answer error";
    const summary = event.message || "推理失败";
    const job = event.job_id ? `任务 ID: ${event.job_id}` : "";
    const detail = event.log_tail ? `详细日志（末尾 12 KB）:\n${event.log_tail}` : "";
    answer.textContent = [summary, job, detail].filter(Boolean).join("\n\n");
    finishRun("失败", true);
  }
}

async function readNdjson(response) {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) consumeEvent(JSON.parse(line));
    }
    if (done) break;
  }
  if (buffer.trim()) consumeEvent(JSON.parse(buffer));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!videoInput.files[0]) return;
  resetRun();
  controller = new AbortController();
  const data = new FormData(form);
  try {
    const response = await fetch("/v1/video/generate/stream", {
      method: "POST",
      body: data,
      signal: controller.signal,
    });
    await readNdjson(response);
  } catch (error) {
    if (error.name === "AbortError") {
      answer.textContent += "\n\n[已停止]";
      finishRun("已停止");
    } else {
      answer.className = "answer error";
      answer.textContent = error.message;
      finishRun("失败", true);
    }
  }
});

cancelButton.addEventListener("click", () => controller?.abort());

fetch("/health")
  .then(async (response) => ({ ok: response.ok, body: await response.json() }))
  .then(({ ok, body }) => {
    const health = document.querySelector("#health");
    health.classList.add(ok ? "ready" : "error");
    document.querySelector("#health-text").textContent = ok
      ? `NPU 就绪 · ${body.profile}`
      : body.issues?.[0] || "服务未就绪";
    runButton.disabled = !ok;
  })
  .catch(() => {
    document.querySelector("#health").classList.add("error");
    document.querySelector("#health-text").textContent = "无法连接服务";
    runButton.disabled = true;
  });
