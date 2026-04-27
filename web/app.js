const transcribeBtn = document.getElementById("transcribeBtn");
const pauseTranscribeBtn = document.getElementById("pauseTranscribeBtn");
const goToSummarizeBtn = document.getElementById("goToSummarizeBtn");
const summarizeBtn = document.getElementById("summarizeBtn");
const cleanSummaryInputBtn = document.getElementById("cleanSummaryInputBtn");
const copyTranscriptBtn = document.getElementById("copyTranscriptBtn");
const downloadTranscriptBtn = document.getElementById("downloadTranscriptBtn");
const copySummaryBtn = document.getElementById("copySummaryBtn");
const downloadSummaryBtn = document.getElementById("downloadSummaryBtn");
const loadConfigBtn = document.getElementById("loadConfigBtn");
const saveConfigBtn = document.getElementById("saveConfigBtn");
const refreshLogsBtn = document.getElementById("refreshLogsBtn");
const copyLogsBtn = document.getElementById("copyLogsBtn");
const clearLogsBtn = document.getElementById("clearLogsBtn");
const sidebarToggleBtn = document.getElementById("sidebarToggleBtn");
const sidebarBackdrop = document.getElementById("sidebarBackdrop");
const resumeFileEl = document.getElementById("resumeFile");
const resumeModeEl = document.getElementById("resumeMode");
const uploadResumeBtn = document.getElementById("uploadResumeBtn");
const resumeStatusEl = document.getElementById("resumeStatus");

const menuButtons = Array.from(document.querySelectorAll(".menuBtn"));
const views = Array.from(document.querySelectorAll(".view"));

const debugModeSettingEl = document.getElementById("debugModeSetting");
const debugPromptEditorsEl = document.getElementById("debugPromptEditors");

const summarySystemPromptEl = document.getElementById("summarySystemPrompt");
const summaryUserPromptTemplateEl = document.getElementById("summaryUserPromptTemplate");
const providerEl = document.getElementById("provider");
const geminiFieldsEl = document.getElementById("geminiFields");
const openaiFieldsEl = document.getElementById("openaiFields");

const transcriptEl = document.getElementById("transcript");
const summaryInputEl = document.getElementById("summaryInput");
const summaryEl = document.getElementById("summary");
const topicEl = document.getElementById("topic");
const debugLogEl = document.getElementById("debugLog");
const statusEl = document.getElementById("status");
const topLogoEl = document.getElementById("topLogo");
const transcribeProgressWrapEl = document.getElementById("transcribeProgressWrap");
const transcribeProgressEl = document.getElementById("transcribeProgress");
const transcribeProgressTextEl = document.getElementById("transcribeProgressText");

const defaultLogoPath = "/Tanukii.png";
const debugLogoPath = "/Tanukii-Light.png";

const localLogs = [];
let debugMode = false;
let lastTranscribeLoggedPercent = null;
let attachedResumeRawFile = null;
let activeTranscribeJobId = "";
let transcribePaused = false;

if (statusEl) {
  statusEl.style.display = "none";
}

function nowLabel() {
  return new Date().toLocaleTimeString("ja-JP", { hour12: false });
}

function setStatus(message) {
  const text = String(message || "").trim();
  if (!text) {
    return;
  }

  const progressMatch = text.match(/文字起こし中\.\.\.\s*(\d+)%/);
  if (progressMatch) {
    const percent = Number(progressMatch[1]);
    if (Number.isFinite(percent) && percent === lastTranscribeLoggedPercent) {
      return;
    }
    lastTranscribeLoggedPercent = percent;
  } else if (/文字起こしが完了しました|文字起こしに失敗しました/.test(text)) {
    lastTranscribeLoggedPercent = null;
  }

  statusEl.style.display = "none";

  let level = "info";
  if (/失敗|エラー|不正|空です|ありません|選択してください/.test(text)) {
    level = "error";
  } else if (/完了|保存しました|読み込みました|更新しました|コピーしました|クリアしました|引き継ぎました/.test(text)) {
    level = "success";
  }

  addLocalLog(level, text);
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function setTranscribeProgress(percent, visible = true) {
  const value = Number.isFinite(percent) ? Math.max(0, Math.min(100, Math.round(percent))) : 0;
  transcribeProgressEl.value = value;
  transcribeProgressTextEl.textContent = `${value}%`;
  transcribeProgressWrapEl.hidden = !visible;
}

async function waitForTranscribeJob(jobId, options = {}) {
  const maxIdleMs = Number.isFinite(options.maxIdleMs) ? options.maxIdleMs : 900000;
  const maxTotalMs = Number.isFinite(options.maxTotalMs) ? options.maxTotalMs : 14400000;
  const onUpdate = typeof options.onUpdate === "function" ? options.onUpdate : null;

  const startedAt = Date.now();
  let lastActivityAt = startedAt;
  let lastJobMessage = "";
  let lastProgress = -1;
  let lastStatus = "";
  let lastUpdatedAt = 0;

  while (true) {
    const now = Date.now();
    if (now - startedAt > maxTotalMs) {
      throw new Error("文字起こし待機が上限時間を超えました。処理を確認してください。");
    }
    if (lastStatus !== "paused" && now - lastActivityAt > maxIdleMs) {
      throw new Error("進捗更新が一定時間なかったため待機を中断しました。");
    }

    const response = await fetchWithTimeout(`/api/transcribe/jobs/${encodeURIComponent(jobId)}`, {}, 15000);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status} ${errorText}`);
    }

    const job = await response.json();
    const status = String(job.status || "");
    const progress = Number.isFinite(job.progress) ? job.progress : 0;
    const updatedAtRaw = Number(job.updated_at);
    const updatedAtMs = Number.isFinite(updatedAtRaw) ? Math.round(updatedAtRaw * 1000) : 0;

    if (status !== lastStatus || progress !== lastProgress || updatedAtMs > lastUpdatedAt) {
      lastActivityAt = now;
      lastStatus = status;
      lastProgress = progress;
      if (updatedAtMs > lastUpdatedAt) {
        lastUpdatedAt = updatedAtMs;
      }
    }

    if (onUpdate) {
      onUpdate(job);
    }

    setTranscribeProgress(progress, true);

    if (status === "completed") {
      setTranscribeProgress(100, true);
      return job.result || {};
    }

    if (status === "failed") {
      throw new Error(job.error || "文字起こしに失敗しました。");
    }

    if (status === "queued") {
      const message = job.message || "文字起こしキュー待機中...";
      if (message !== lastJobMessage) {
        setStatus(message);
        lastJobMessage = message;
      }
    } else {
      const message = job.message || `文字起こし中... ${Math.round(progress)}%`;
      if (message !== lastJobMessage) {
        setStatus(message);
        lastJobMessage = message;
      }
    }

    await sleep(700);
  }
}

function syncPauseButton() {
  if (!pauseTranscribeBtn) {
    return;
  }
  const hasActiveJob = !!activeTranscribeJobId;
  pauseTranscribeBtn.disabled = !hasActiveJob;
  pauseTranscribeBtn.textContent = transcribePaused ? "再開" : "一時停止";
}

function clearTranscribeJobState() {
  activeTranscribeJobId = "";
  transcribePaused = false;
  syncPauseButton();
}

async function toggleTranscribePause() {
  if (!activeTranscribeJobId) {
    return;
  }

  const action = transcribePaused ? "resume" : "pause";
  pauseTranscribeBtn.disabled = true;

  try {
    const response = await fetchWithTimeout(
      `/api/transcribe/jobs/${encodeURIComponent(activeTranscribeJobId)}/${action}`,
      { method: "POST" },
      10000
    );
    if (!response.ok) {
      throw new Error(await readErrorDetail(response));
    }

    transcribePaused = !transcribePaused;
    syncPauseButton();
    setStatus(transcribePaused ? "文字起こしを一時停止しました。" : "文字起こしを再開しました。");
  } catch (error) {
    console.error(error);
    logDebug("error", `transcribe ${action} failed: ${error.message}`);
    setStatus(transcribePaused ? "文字起こしの再開に失敗しました。" : "文字起こしの一時停止に失敗しました。");
  } finally {
    syncPauseButton();
  }
}

function addLocalLog(level, message) {
  const line = `[${nowLabel()}] ${level.toUpperCase()} ${message}`;
  localLogs.push(line);
  if (localLogs.length > 400) {
    localLogs.splice(0, localLogs.length - 400);
  }
  debugLogEl.value = localLogs.join("\n");
  debugLogEl.scrollTop = debugLogEl.scrollHeight;
}

function logDebug(level, message) {
  if (!debugMode) {
    return;
  }
  addLocalLog(level, message);
}

function showView(viewId) {
  for (const view of views) {
    view.classList.toggle("active", view.id === viewId);
  }
  for (const button of menuButtons) {
    button.classList.toggle("active", button.dataset.view === viewId);
  }
}

function setSidebarOpen(open) {
  document.body.classList.toggle("sidebar-open", !!open);
  sidebarToggleBtn.textContent = open ? "✕" : "☰";
  sidebarToggleBtn.setAttribute("aria-expanded", open ? "true" : "false");
}

function toggleSidebar() {
  setSidebarOpen(!document.body.classList.contains("sidebar-open"));
}

function setDebugMode(enabled) {
  debugMode = !!enabled;
  document.body.classList.toggle("debug-active", debugMode);
  debugModeSettingEl.checked = debugMode;
  debugPromptEditorsEl.style.display = debugMode ? "block" : "none";

  if (topLogoEl) {
    topLogoEl.src = debugMode ? debugLogoPath : defaultLogoPath;
  }

  const menuDebug = document.getElementById("menuDebug");
  if (!debugMode && menuDebug.classList.contains("active")) {
    showView("transcribeView");
  }

  if (debugMode) {
    addLocalLog("info", "debug mode enabled");
  }
}

function updateProviderFields() {
  const provider = providerEl.value;
  geminiFieldsEl.style.display = provider === "gemini" ? "block" : "none";
  openaiFieldsEl.style.display = provider === "openai" ? "block" : "none";
}

function readConfigFromInputs() {
  return {
    whisper_model: document.getElementById("whisperModel").value || "small",
    debug_mode: !!debugModeSettingEl.checked,
    summary_system_prompt: summarySystemPromptEl.value || "",
    summary_user_prompt_template: summaryUserPromptTemplateEl.value || "",
    summary_provider: providerEl.value || "auto",
    openai_model: document.getElementById("openaiModel").value || "",
    openai_api_key: document.getElementById("openaiApiKey").value || "",
    gemini_model: document.getElementById("geminiModel").value || "",
    gemini_api_key: document.getElementById("geminiApiKey").value || "",
  };
}

function applyConfigToInputs(config) {
  if (!config) {
    return;
  }

  document.getElementById("whisperModel").value = config.whisper_model || "small";
  providerEl.value = config.summary_provider || "auto";
  document.getElementById("openaiModel").value = config.openai_model || "";
  document.getElementById("openaiApiKey").value = config.openai_api_key || "";
  document.getElementById("geminiModel").value = config.gemini_model || "";
  document.getElementById("geminiApiKey").value = config.gemini_api_key || "";
  summarySystemPromptEl.value = config.summary_system_prompt || "";
  summaryUserPromptTemplateEl.value = config.summary_user_prompt_template || "";

  setDebugMode(!!config.debug_mode);
  updateProviderFields();
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 300000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function readErrorDetail(response) {
  let bodyText = "";
  try {
    bodyText = (await response.text()) || "";
  } catch {
    return `HTTP ${response.status}`;
  }

  if (!bodyText.trim()) {
    return `HTTP ${response.status}`;
  }

  try {
    const parsed = JSON.parse(bodyText);
    if (parsed && typeof parsed.detail === "string" && parsed.detail.trim()) {
      return `HTTP ${response.status} ${parsed.detail}`;
    }
  } catch {
    // fall through and return raw text
  }

  return `HTTP ${response.status} ${bodyText}`;
}

async function loadConfig() {
  try {
    const response = await fetchWithTimeout("/api/config", {}, 10000);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const config = await response.json();
    applyConfigToInputs(config);
    setStatus("設定を読み込みました。");
  } catch (error) {
    console.error(error);
    logDebug("error", `config load failed: ${error.message}`);
    setStatus("設定の読み込みに失敗しました。");
  }
}

async function saveConfig() {
  try {
    const body = readConfigFromInputs();
    const response = await fetchWithTimeout(
      "/api/config",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
      10000
    );
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    setStatus("設定を保存しました。");
  } catch (error) {
    console.error(error);
    logDebug("error", `config save failed: ${error.message}`);
    setStatus("設定の保存に失敗しました。");
  }
}

async function refreshDebugLogs() {
  try {
    const response = await fetchWithTimeout("/api/debug/logs", {}, 10000);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const serverLines = (data.logs || []).map(
      (entry) => `[${entry.ts}] ${String(entry.level || "info").toUpperCase()} ${entry.message || ""}`
    );
    const merged = ["=== Client Logs ===", ...localLogs, "", "=== Server Logs ===", ...serverLines];
    debugLogEl.value = merged.join("\n");
    debugLogEl.scrollTop = debugLogEl.scrollHeight;
    setStatus("ログを更新しました。");
  } catch (error) {
    console.error(error);
    logDebug("error", `debug refresh failed: ${error.message}`);
    setStatus("ログ更新に失敗しました。");
  }
}

async function clearDebugLogs() {
  try {
    await fetchWithTimeout("/api/debug/logs/clear", { method: "POST" }, 10000);
    localLogs.length = 0;
    debugLogEl.value = "";
    setStatus("ログをクリアしました。");
  } catch (error) {
    console.error(error);
    logDebug("error", `debug clear failed: ${error.message}`);
    setStatus("ログクリアに失敗しました。");
  }
}

async function copyDebugLogs() {
  try {
    await navigator.clipboard.writeText(debugLogEl.value || "");
    setStatus("ログをコピーしました。");
  } catch (error) {
    console.error(error);
    logDebug("error", `debug copy failed: ${error.message}`);
    setStatus("ログコピーに失敗しました。");
  }
}

async function copyTextResult(text, successMessage, emptyMessage) {
  if (!text.trim()) {
    setStatus(emptyMessage);
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    setStatus(successMessage);
  } catch (error) {
    console.error(error);
    setStatus("コピーに失敗しました。");
  }
}

function sanitizeFilename(filename, fallbackName) {
  const trimmed = String(filename || "").trim();
  if (!trimmed) {
    return fallbackName;
  }

  const normalized = trimmed
    .replace(/[\\/:*?"<>|]/g, "_")
    .replace(/\s+/g, " ")
    .trim();

  return normalized || fallbackName;
}

async function saveTextWithPicker(text, filename) {
  if (typeof window.showSaveFilePicker !== "function") {
    return false;
  }

  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const handle = await window.showSaveFilePicker({
    suggestedName: filename,
    types: [
      {
        description: "Text",
        accept: {
          "text/plain": [".txt"],
        },
      },
    ],
  });

  const writable = await handle.createWritable();
  await writable.write(blob);
  await writable.close();
  return true;
}

async function downloadTextResult(text, filename, emptyMessage) {
  if (!text.trim()) {
    setStatus(emptyMessage);
    return;
  }
  const selectedFilename = sanitizeFilename(filename, filename);

  try {
    const savedWithPicker = await saveTextWithPicker(text, selectedFilename);
    if (savedWithPicker) {
      setStatus("保存先を選択して保存しました。");
      return;
    }

    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = selectedFilename;
    anchor.click();
    URL.revokeObjectURL(url);
    setStatus("既定のファイル名で保存しました。");
  } catch (error) {
    if (error && error.name === "AbortError") {
      setStatus("保存をキャンセルしました。");
      return;
    }
    console.error(error);
    setStatus("保存に失敗しました。");
  }
}

function forwardToSummarize() {
  summaryInputEl.value = transcriptEl.value;
  showView("summarizeView");
  setStatus("文字起こし結果を要約に引き継ぎました。");
}

function cleanSummaryInputText() {
  const originalText = summaryInputEl.value || "";
  if (!originalText.trim()) {
    setStatus("要約対象テキストが空です。");
    return;
  }

  const fillers = [
    "え、",
    "え,",
    "えー、",
    "えー,",
    "えーと、",
    "えーと,",
    "えっと、",
    "えっと,",
    "ま、",
    "ま,",
    "まあ、",
    "まあ,",
    "まぁ、",
    "まぁ,",
    "あ、",
    "あ,",
    "あの、",
    "あの,",
    "あー、",
    "あー,",
    "うーん、",
    "うーん,",
    "うーんと、",
    "うーんと,",
    "なんか、",
    "なんか,",
    "そのー、",
    "そのー,",
    "あのー、",
    "あのー,",
    "うん。"
  ];

  let cleaned = originalText.replace(/\r\n/g, "\n");
  cleaned = cleaned.replace(/(^|[^\n。．\.！？!?])\n+/g, "$1");
  cleaned = cleaned.replace(/[ \u3000]+/g, "");
  for (const filler of fillers) {
    cleaned = cleaned.split(filler).join("");
  }

  summaryInputEl.value = cleaned;
  setStatus(`要約対象テキストを整形しました。(${originalText.length} → ${cleaned.length}文字)`);
}

async function transcribe() {
  const fileInput = document.getElementById("audio");
  const languageInput = document.getElementById("language");
  const topicInput = document.getElementById("topic");
  const whisperModelInput = document.getElementById("whisperModel");
  const file = fileInput.files[0];

  if (!file) {
    setStatus("音声ファイルを選択してください。");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("language", languageInput.value || "ja");
  formData.append("model", whisperModelInput.value || "small");
  formData.append("initial_prompt", topicInput.value || "");

  setTranscribeProgress(0, true);
  clearTranscribeJobState();
  setStatus("文字起こしジョブを開始します...");
  logDebug("info", `transcribe requested: file=${file.name}, model=${whisperModelInput.value}, has_topic=${!!topicInput.value}`);
  transcribeBtn.disabled = true;

  try {
    const response = await fetchWithTimeout(
      "/api/transcribe/start",
      { method: "POST", body: formData },
      60000
    );
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status} ${errorText}`);
    }

    const started = await response.json();
    if (!started.job_id) {
      throw new Error("job_id が返されませんでした。");
    }

    logDebug("info", `transcribe job started: job=${started.job_id}`);
    activeTranscribeJobId = started.job_id;
    transcribePaused = false;
    syncPauseButton();
    const data = await waitForTranscribeJob(started.job_id, {
      maxIdleMs: 900000,
      maxTotalMs: 14400000,
      onUpdate: (job) => {
        const paused = job.status === "paused" || !!job.pause_requested;
        if (paused !== transcribePaused) {
          transcribePaused = paused;
          syncPauseButton();
        }
      },
    });
    transcriptEl.value = data.text || "";
    summaryInputEl.value = data.text || "";
    logDebug("info", `transcribe done: chars=${(data.text || "").length}`);
    setStatus("文字起こしが完了しました。");
  } catch (error) {
    console.error(error);
    logDebug("error", `transcribe failed: ${error.message}`);
    setTranscribeProgress(0, false);
    setStatus("文字起こしに失敗しました。");
  } finally {
    clearTranscribeJobState();
    transcribeBtn.disabled = false;
  }
}

async function summarize() {
  const style = document.getElementById("style").value;
  const provider = providerEl.value;
  const resumeMode = resumeModeEl.value || "extract";
  const inputText = summaryInputEl.value;
  const rawFileForRequest = resumeMode === "raw"
    ? (attachedResumeRawFile || resumeFileEl.files[0] || null)
    : null;
  const canSummarizeWithRawOnly = !!rawFileForRequest;

  if (!inputText.trim() && !canSummarizeWithRawOnly) {
    setStatus("要約対象テキストが空です。");
    return;
  }

  const model = provider === "openai"
    ? document.getElementById("openaiModel").value
    : document.getElementById("geminiModel").value;
  const apiKey = provider === "openai"
    ? document.getElementById("openaiApiKey").value
    : document.getElementById("geminiApiKey").value;

  const formData = new FormData();
  formData.append("text", inputText);
  formData.append("style", style);
  formData.append("provider", provider);
  formData.append("resume_mode", rawFileForRequest ? "raw" : resumeMode);
  formData.append("model", model);
  formData.append("api_key", apiKey);

  if (rawFileForRequest) {
    formData.append("resume_file", rawFileForRequest, rawFileForRequest.name);
  }

  if (debugMode) {
    formData.append("system_prompt", summarySystemPromptEl.value || "");
    formData.append("user_prompt_template", summaryUserPromptTemplateEl.value || "");
  }

  setStatus("要約を生成中...");
  summarizeBtn.disabled = true;
  logDebug("info", `summarize requested: provider=${provider}, style=${style}`);

  try {
    const response = await fetchWithTimeout(
      "/api/summarize",
      { method: "POST", body: formData },
      120000
    );
    if (!response.ok) {
      throw new Error(await readErrorDetail(response));
    }

    const data = await response.json();
    summaryEl.value = data.summary || "";
    setStatus(`要約が完了しました (${data.summary_mode || "unknown"})。`);
    logDebug("info", `summarize done: mode=${data.summary_mode || "unknown"}`);
  } catch (error) {
    console.error(error);
    logDebug("error", `summarize failed: ${error.message}`);
    setStatus(`要約に失敗しました。(${error.message})`);
  } finally {
    summarizeBtn.disabled = false;
  }
}

async function uploadResume() {
  const file = resumeFileEl.files[0];
  const resumeMode = resumeModeEl.value || "extract";
  if (!file) {
    setStatus("レジュメファイルを選択してください。");
    return;
  }

  const allowedTypes = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  ];
  const fileNameLower = (file.name || "").toLowerCase();
  const allowedByExt = [".pdf", ".docx", ".pptx"].some((ext) => fileNameLower.endsWith(ext));
  if (!allowedTypes.includes(file.type) && !allowedByExt) {
    setStatus("PDF / Word (.docx) / PowerPoint (.pptx) のみ対応しています。");
    return;
  }

  resumeStatusEl.style.display = "block";
  resumeStatusEl.innerHTML = "レジュメを読み込み中...";
  uploadResumeBtn.disabled = true;

  try {
    if (resumeMode === "raw") {
      attachedResumeRawFile = file;
      resumeStatusEl.innerHTML = `✓ RAW添付を保持: ${file.name}`;
      setStatus(`RAWモードでレジュメを保持しました。(${file.name})`);
      logDebug("info", `resume raw attached: filename=${file.name}, bytes=${file.size}`);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetchWithTimeout(
      "/api/extract-resume",
      { method: "POST", body: formData },
      60000
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status} ${errorText}`);
    }

    const data = await response.json();
    const resumeText = data.text || "";
    const currentText = summaryInputEl.value.trim();
    attachedResumeRawFile = null;

    if (currentText) {
      summaryInputEl.value = currentText + "\n\n【レジュメ内容】\n" + resumeText;
    } else {
      summaryInputEl.value = resumeText;
    }

    resumeStatusEl.innerHTML = `✓ 読み込み完了: ${file.name}`;
    setStatus(`レジュメを読み込みました。(${file.name})`);
    logDebug("info", `resume loaded: filename=${file.name}, chars=${resumeText.length}`);
  } catch (error) {
    console.error(error);
    resumeStatusEl.innerHTML = `✗ エラー: ${error.message}`;
    logDebug("error", `resume upload failed: ${error.message}`);
    setStatus("レジュメの読み込みに失敗しました。");
  } finally {
    uploadResumeBtn.disabled = false;
  }
}

if (resumeModeEl) {
  resumeModeEl.addEventListener("change", () => {
    if (resumeModeEl.value !== "raw") {
      attachedResumeRawFile = null;
    }
  });
}

for (const button of menuButtons) {
  button.addEventListener("click", () => {
    const viewId = button.dataset.view;
    if (viewId === "debugView" && !debugMode) {
      setStatus("デバッグモードを設定で有効化してください。");
      showView("settingsView");
      return;
    }
    showView(viewId);
    if (window.innerWidth <= 980) {
      setSidebarOpen(false);
    }
  });
}

providerEl.addEventListener("change", updateProviderFields);
debugModeSettingEl.addEventListener("change", () => setDebugMode(debugModeSettingEl.checked));
transcribeBtn.addEventListener("click", transcribe);
pauseTranscribeBtn.addEventListener("click", toggleTranscribePause);
goToSummarizeBtn.addEventListener("click", forwardToSummarize);
summarizeBtn.addEventListener("click", summarize);
cleanSummaryInputBtn.addEventListener("click", cleanSummaryInputText);
uploadResumeBtn.addEventListener("click", uploadResume);
loadConfigBtn.addEventListener("click", loadConfig);
saveConfigBtn.addEventListener("click", saveConfig);
refreshLogsBtn.addEventListener("click", refreshDebugLogs);
copyLogsBtn.addEventListener("click", copyDebugLogs);
clearLogsBtn.addEventListener("click", clearDebugLogs);
copyTranscriptBtn.addEventListener("click", () =>
  copyTextResult(transcriptEl.value, "文字起こし結果をコピーしました。", "コピーする文字起こし結果がありません。")
);
downloadTranscriptBtn.addEventListener("click", () =>
  downloadTextResult(transcriptEl.value, "transcript.txt", "保存する文字起こし結果がありません。")
);
copySummaryBtn.addEventListener("click", () =>
  copyTextResult(summaryEl.value, "要約結果をコピーしました。", "コピーする要約結果がありません。")
);
downloadSummaryBtn.addEventListener("click", () =>
  downloadTextResult(summaryEl.value, "summary.txt", "保存する要約結果がありません。")
);
sidebarToggleBtn.addEventListener("click", toggleSidebar);
sidebarBackdrop.addEventListener("click", () => setSidebarOpen(false));
window.addEventListener("resize", () => {
  if (window.innerWidth > 980) {
    setSidebarOpen(true);
  }
});

setDebugMode(false);
updateProviderFields();
showView("transcribeView");
setSidebarOpen(window.innerWidth > 980);
loadConfig();
syncPauseButton();
