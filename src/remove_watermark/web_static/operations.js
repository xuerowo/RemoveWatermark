import {
  IMAGE_FILE_EXTENSIONS,
  app,
  batchProgress,
  batchProgressFill,
  batchProgressLabel,
  batchProgressMeta,
  processingTime,
  statusText,
} from "./state.js";
import { api, downloadUrl, imageUrl } from "./api.js";
import {
  buildMaskSurface,
  drawImageToSurface,
  loadImage,
  renderCanvas,
  updateCursorCoordinates,
  updateMaskTint,
  updateZoomLimit,
} from "./canvas.js";
import {
  applySummary,
  clearCurrentImageState,
  imageHasResettableState,
  imageIdentity,
  renderDetectionList,
  renderImageList,
  renderTemplateList,
  setRenderActions,
  updateButtons,
} from "./render.js";
import { detectionRequestPayload, processRequestPayload } from "./settings.js";

export function setStatus(message) {
  statusText.textContent = message;
}

export function setProcessingTime(seconds = null) {
  processingTime.textContent = seconds === null ? "處理時間 --" : `處理時間 ${formatDuration(seconds)}`;
}

export function elapsedSince(startedAt) {
  return (performance.now() - startedAt) / 1000;
}

export function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return minutes ? `${minutes} 分 ${String(rest).padStart(2, "0")} 秒` : `${rest} 秒`;
}

export function setBusy(value) {
  app.busy = value;
  updateButtons();
}

export function updateBatchProgress(percent, label, meta, running = true) {
  batchProgress.hidden = false;
  batchProgressLabel.textContent = label;
  batchProgressMeta.textContent = meta;
  batchProgressFill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
  batchProgressFill.classList.toggle("running", running);
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function currentProgressTotal() {
  return Number(app.batchProgressLatest?.totalImages || app.batchProgressTotal || 0);
}

export function displayProgressKind(kind, total = currentProgressTotal()) {
  return Number(total) === 1 && kind.startsWith("批量") ? kind.slice(2) : kind;
}

export function beginBatchProgress(kind, total, estimated = true) {
  clearInterval(app.batchProgressTimer);
  clearTimeout(app.batchProgressHideTimer);
  app.batchProgressHideTimer = null;
  app.batchProgressKind = kind;
  app.batchProgressTotal = total;
  app.batchProgressStartedAt = Date.now();
  setProcessingTime(null);
  app.batchProgressLatest = null;
  app.batchProgressAppliedKey = "";
  setBusy(true);
  const labelKind = displayProgressKind(kind, total);
  updateBatchProgress(4, `${labelKind}準備中`, `共 ${total} 張`, true);
  if (!estimated) return;
  app.batchProgressTimer = setInterval(() => {
    const elapsed = (Date.now() - app.batchProgressStartedAt) / 1000;
    const percent = Math.min(92, 8 + Math.log1p(elapsed) * 20);
    updateBatchProgress(percent, `${labelKind}處理中`, `${formatDuration(elapsed)} · 共 ${total} 張`, true);
  }, 500);
}

export function updateTrackedBatchProgress(kind, job) {
  app.batchProgressLatest = job;
  const total = Number(job.totalImages || 0);
  const processed = Number(job.processedImages || 0);
  const failed = Number(job.failedImages || 0);
  const remaining = Number(job.remainingImages || Math.max(0, total - processed - failed));
  const done = processed + failed;
  const elapsed = (Date.now() - app.batchProgressStartedAt) / 1000;
  const percent = total > 0 ? (done / total) * 100 : 100;
  const labelKind = displayProgressKind(kind, total);
  const label = job.status === "completed" ? `${labelKind}完成` : `${labelKind}處理中`;
  const meta = `已完成 ${processed} / 剩 ${remaining} / 失敗 ${failed} · ${formatDuration(elapsed)}`;
  updateBatchProgress(job.status === "completed" ? 100 : percent, label, meta, job.status === "running");
}

export function liveSummaryKey(summary) {
  return (summary.images || []).map((image) => [
    image.index,
    image.status || "",
    image.maskVersion || "",
    image.resultVersion || "",
    (image.detections || []).length,
    image.error || "",
  ].join(":")).join("|");
}

export function imageChanged(previous, next) {
  if (!previous || !next) return Boolean(previous || next);
  return previous.status !== next.status
    || previous.maskVersion !== next.maskVersion
    || previous.resultVersion !== next.resultVersion
    || previous.error !== next.error
    || (previous.detections || []).length !== (next.detections || []).length;
}

export function mergeLiveSummaryImages(summary) {
  if (!summary?.images || !app.images.length) return summary;
  const summaryByKey = new Map();
  summary.images.forEach((image) => {
    const key = imageIdentity(image);
    if (key) summaryByKey.set(key, image);
  });
  if (!summaryByKey.size) return summary;

  const seen = new Set();
  const mergedImages = app.images.map((current) => {
    const currentKey = imageIdentity(current);
    const updated = currentKey ? summaryByKey.get(currentKey) : null;
    if (!updated) return current;
    seen.add(currentKey);
    return { ...updated, index: current.index };
  });
  summary.images.forEach((image) => {
    const key = imageIdentity(image);
    if (!key || !seen.has(key)) mergedImages.push(image);
  });
  return { ...summary, images: mergedImages };
}

export async function applyLiveBatchSummary(summary, { showMaskOverlay = false } = {}) {
  if (!summary?.images) return;
  const mergedSummary = mergeLiveSummaryImages(summary);
  const key = liveSummaryKey(mergedSummary);
  if (!key || key === app.batchProgressAppliedKey) return;
  app.batchProgressAppliedKey = key;
  const previousImage = app.images[app.currentIndex];
  applySummary(mergedSummary);
  const index = app.currentIndex;
  const nextImage = app.images[index];
  if (imageChanged(previousImage, nextImage)) {
    const loaded = await loadCurrentImage(index, false);
    if (loaded && app.currentIndex === index) {
      app.showMaskOverlay = Boolean(showMaskOverlay && nextImage?.hasMask && !nextImage?.hasResult);
      renderCanvas();
    }
  }
}

export async function waitForBatchJob(kind, jobId, options = {}) {
  app.activeBatchJobId = jobId;
  while (true) {
    const payload = await api(`/api/batch-progress?id=${encodeURIComponent(jobId)}`);
    const job = payload.job || {};
    updateTrackedBatchProgress(kind, job);
    await applyLiveBatchSummary(job.summary, options);
    if (job.status === "completed") {
      app.activeBatchJobId = "";
      return job.summary;
    }
    if (job.status === "failed") {
      app.activeBatchJobId = "";
      throw new Error(job.error || "作業失敗");
    }
    if (job.status === "cancelled") {
      app.activeBatchJobId = "";
      throw new Error(job.error || "作業已中斷");
    }
    await sleep(500);
  }
}

export function finishBatchProgress(kind, summaryText, failed) {
  clearInterval(app.batchProgressTimer);
  clearTimeout(app.batchProgressHideTimer);
  app.batchProgressTimer = null;
  app.activeBatchJobId = "";
  const elapsed = (Date.now() - app.batchProgressStartedAt) / 1000;
  const labelKind = displayProgressKind(kind);
  const label = failed ? `${labelKind}完成，有失敗` : `${labelKind}完成`;
  updateBatchProgress(100, label, `${summaryText} · ${formatDuration(elapsed)}`, false);
  setProcessingTime(elapsed);
  app.operationStopRequested = false;
  setBusy(false);
  app.batchProgressHideTimer = setTimeout(() => {
    if (!app.batchProgressTimer) batchProgress.hidden = true;
    app.batchProgressHideTimer = null;
  }, 4500);
}

export function failBatchProgress(kind, message) {
  clearInterval(app.batchProgressTimer);
  clearTimeout(app.batchProgressHideTimer);
  app.batchProgressTimer = null;
  app.activeBatchJobId = "";
  app.batchProgressHideTimer = null;
  const elapsed = app.batchProgressStartedAt ? (Date.now() - app.batchProgressStartedAt) / 1000 : 0;
  updateBatchProgress(100, `${displayProgressKind(kind)}失敗`, `${formatDuration(elapsed)} · ${message}`, false);
  setProcessingTime(elapsed);
  app.operationStopRequested = false;
  setBusy(false);
}

export function cancelBatchProgress(kind, message = "已中斷") {
  clearInterval(app.batchProgressTimer);
  clearTimeout(app.batchProgressHideTimer);
  app.batchProgressTimer = null;
  app.activeBatchJobId = "";
  app.batchProgressHideTimer = null;
  const elapsed = app.batchProgressStartedAt ? (Date.now() - app.batchProgressStartedAt) / 1000 : 0;
  updateBatchProgress(100, `${displayProgressKind(kind)}已中斷`, `${formatDuration(elapsed)} · ${message}`, false);
  setProcessingTime(elapsed);
  app.operationStopRequested = false;
  setBusy(false);
}

export function operationKindLabel(mode) {
  return {
    detect: "批量偵測",
    process: "批量去水印",
    detectProcess: "批量一鍵去水印",
  }[mode] || "批量處理";
}

export function operationButtonLabel(mode) {
  return {
    detect: "偵測",
    process: "去水印",
    detectProcess: "一鍵去水印",
  }[mode] || "處理";
}

export function hasOperationBatchItem(index, mode) {
  const key = imageIdentity(app.images[index]);
  return Boolean(key) && app.operationMode === mode && app.operationPaths.has(key);
}

export function canStartOperation(mode) {
  if (!app.images.length) return false;
  const currentImage = app.images[app.currentIndex];
  const needsTemplate = app.detectorMode === "template";
  if ((mode === "detect" || mode === "detectProcess") && needsTemplate && app.selectedTemplates.size === 0) return false;
  if (mode === "process" && (!currentImage?.hasMask || !app.maskCanvas || app.loadedImageIndex !== app.currentIndex)) return false;
  return true;
}

export function buildOperationPayload(mode) {
  const index = app.currentIndex;
  const image = app.images[index];
  const imagePath = image?.path || "";
  const imageKey = imageIdentity(image);
  if (mode === "detect") {
    return { mode, index, imagePath, imageKey, ...detectionRequestPayload() };
  }
  if (mode === "process") {
    if (app.loadedImageIndex !== index || !app.maskCanvas) {
      throw new Error("目前圖片還沒載入完成，請稍後再處理");
    }
    return {
      mode,
      index,
      imagePath,
      imageKey,
      maskData: app.maskCanvas.toDataURL("image/png"),
      ...processRequestPayload(),
    };
  }
  return {
    mode,
    index,
    imagePath,
    imageKey,
    ...detectionRequestPayload(),
    ...processRequestPayload(),
  };
}

export function operationSummaryText(mode, summary) {
  const batch = summary.batch || {};
  const failed = Number(batch.failedImages || 0);
  const processed = Number(batch.processedImages || 0);
  if (mode === "detect" || mode === "detectProcess") {
    const detections = Number(batch.detectionCount || 0);
    return `${processed} 張，${detections} 個區域${failed ? `，${failed} 張失敗` : ""}`;
  }
  return `${processed} 張${failed ? `，${failed} 張失敗` : ""}`;
}

export async function monitorOperationBatch(mode, jobId) {
  const kind = operationKindLabel(mode);
  try {
    const summary = await waitForBatchJob(kind, jobId, { showMaskOverlay: mode === "detect" });
    if (!summary) throw new Error("工作沒有回傳結果");
    applySummary(summary);
    await loadCurrentImage(Math.min(app.currentIndex, app.images.length - 1), false);
    app.showMaskOverlay = mode === "detect" && app.images[app.currentIndex]?.hasMask && !app.images[app.currentIndex]?.hasResult;
    renderCanvas();
    const failed = Number(summary.batch?.failedImages || 0);
    const message = operationSummaryText(mode, summary);
    app.operationMode = "";
    app.operationPaths = new Set();
    finishBatchProgress(kind, message, failed > 0);
    setStatus(`${displayProgressKind(kind)}完成：${message}`);
  } catch (error) {
    app.operationMode = "";
    app.operationPaths = new Set();
    if (app.operationStopRequested || error.message === "作業已中斷") {
      cancelBatchProgress(kind);
      setStatus(`已中斷${displayProgressKind(kind)}`);
    } else {
      failBatchProgress(kind, error.message);
      setStatus(error.message);
    }
  } finally {
    app.operationMode = "";
    app.operationPaths = new Set();
    app.operationStopRequested = false;
    updateButtons();
  }
}

export async function startOrAddCurrentOperation(mode) {
  if (!canStartOperation(mode)) return;
  if (app.operationStopRequested) {
    setStatus("正在中斷處理，請等待目前工作停止");
    return;
  }
  if (app.operationMode && app.operationMode !== mode) {
    setStatus(`目前正在${operationButtonLabel(app.operationMode)}，只能加入同類型處理`);
    return;
  }
  if (hasOperationBatchItem(app.currentIndex, mode)) {
    setStatus(`第 ${app.currentIndex + 1} 張已在目前工作中`);
    return;
  }
  if (app.busy && !app.operationMode) {
    setStatus("目前正在處理中，請先中斷或等待完成");
    return;
  }

  const wasIdle = !app.operationMode;
  let openedProgress = false;
  try {
    const payload = buildOperationPayload(mode);
    if (wasIdle) {
      app.operationMode = mode;
      app.operationStopRequested = false;
      app.operationPaths = new Set([payload.imageKey]);
      beginBatchProgress(operationKindLabel(mode), 1, false);
      openedProgress = true;
      const started = await api("/api/start-operation-batch", payload);
      const job = started.job || {};
      if (!job.id) throw new Error("工作沒有取得工作編號");
      app.activeBatchJobId = job.id;
      app.operationPaths = new Set(job.itemPaths || [payload.imageKey]);
      if (app.operationStopRequested) {
        await cancelServerProcessing({ jobId: job.id });
        throw new Error("作業已中斷");
      }
      updateTrackedBatchProgress(operationKindLabel(mode), job);
      setStatus(`開始${displayProgressKind(operationKindLabel(mode), 1)}：第 ${payload.index + 1} 張`);
      updateButtons();
      void monitorOperationBatch(mode, job.id);
      return;
    }

    const previousOperationPaths = new Set(app.operationPaths);
    app.operationPaths.add(payload.imageKey);
    updateButtons();
    let added;
    try {
      added = await api("/api/add-operation-batch", {
        jobId: app.activeBatchJobId,
        ...payload,
      });
    } catch (error) {
      app.operationPaths = previousOperationPaths;
      updateButtons();
      throw error;
    }
    const job = added.job || {};
    app.operationPaths = new Set(job.itemPaths || [...app.operationPaths, payload.imageKey]);
    updateTrackedBatchProgress(operationKindLabel(mode), job);
    setStatus(`已加入${displayProgressKind(operationKindLabel(mode), job.totalImages || 2)}：第 ${payload.index + 1} 張`);
    updateButtons();
  } catch (error) {
    if (wasIdle) {
      app.operationMode = "";
      app.operationPaths = new Set();
      if (openedProgress) {
        failBatchProgress(operationKindLabel(mode), error.message);
      } else {
        setBusy(false);
      }
    }
    setStatus(error.message);
  }
}

export async function cancelServerProcessing(payload) {
  try {
    await api("/api/cancel-processing", payload);
  } catch {
    // 中斷是盡力而為；後端批量工作仍會停止後續項目。
  }
}

export async function stopAllProcessing() {
  if (!app.busy) return;
  app.operationStopRequested = true;
  setStatus("正在中斷處理...");
  if (app.activeBatchJobId) {
    void cancelServerProcessing({ jobId: app.activeBatchJobId });
  }
  updateButtons();
}

export async function refreshState() {
  const state = await api("/api/state");
  app.images = state.images;
  app.templates = state.templates;
  applySummary(state);
  if (app.selectedTemplates.size === 0) {
    app.templates.forEach((template) => app.selectedTemplates.add(template.path));
  }
  renderImageList();
  renderTemplateList();
  if (app.images.length) {
    await loadCurrentImage(Math.min(app.currentIndex, app.images.length - 1), true);
  } else {
    clearCurrentImageState("尚未新增圖片");
    setStatus(state.inputIsTemporary ? "請新增圖片或匯入資料夾" : "找不到可載入的圖片");
  }
}

export async function loadCurrentOrEmpty(index, emptyMessage = "尚未載入圖片") {
  if (app.images.length) {
    await loadCurrentImage(Math.min(Math.max(index, 0), app.images.length - 1), true);
  } else {
    clearCurrentImageState(emptyMessage);
  }
}

export async function loadCurrentImage(index, resetView) {
  if (!app.images[index]) return false;
  const requestId = ++app.loadRequestId;
  app.currentIndex = index;
  renderImageList();
  renderDetectionList();
  updateButtons();
  const imageState = app.images[index];
  const [original, result, mask] = await Promise.all([
    loadImage(imageUrl(index, "original")),
    loadImage(imageUrl(index, "result")),
    loadImage(imageUrl(index, "mask")),
  ]);
  if (requestId !== app.loadRequestId || app.currentIndex !== index || !app.images[index]) {
    return false;
  }
  const width = original.naturalWidth;
  const height = original.naturalHeight;
  app.originalCanvas = drawImageToSurface(original, width, height);
  app.resultCanvas = drawImageToSurface(result, width, height);
  app.maskCanvas = buildMaskSurface(mask, width, height);
  app.loadedImageIndex = index;
  updateMaskTint();
  if (resetView) {
    app.zoom = 1;
    app.pan = { x: 0, y: 0 };
    app.showMaskOverlay = imageState.hasMask && !imageState.hasResult && (imageState.status === "detected" || imageState.status === "edited");
  }
  updateZoomLimit();
  emptyState.style.display = "none";
  document.getElementById("imageMeta").textContent = `${app.images[index].name} · ${width} x ${height}`;
  updateCursorCoordinates();
  renderDetectionList();
  updateButtons();
  renderCanvas();
  return true;
}

export async function persistMask(showOverlay = true) {
  if (!app.maskCanvas || !app.images[app.currentIndex]) return;
  try {
    const updated = await api("/api/save-mask", {
      index: app.currentIndex,
      maskData: app.maskCanvas.toDataURL("image/png"),
      detections: app.images[app.currentIndex].detections || [],
    });
    app.images[app.currentIndex] = updated;
    app.showMaskOverlay = showOverlay;
    renderImageList();
    renderCanvas();
    updateButtons();
  } catch (error) {
    setStatus(error.message);
  }
}

export async function clearDetections() {
  if (!app.images.length || !app.maskCanvas) return;
  try {
    const updated = await api("/api/save-mask", {
      index: app.currentIndex,
      maskData: app.maskCanvas.toDataURL("image/png"),
      detections: [],
    });
    app.images[app.currentIndex] = updated;
    renderImageList();
    renderDetectionList();
    renderCanvas();
    updateButtons();
    setStatus("已清除目前圖片的偵測框");
  } catch (error) {
    setStatus(error.message);
  }
}

export async function clearAllMasks() {
  if (!app.images.length) return;
  if (!window.confirm("要清除所有圖片的遮罩嗎？偵測框與結果圖會保留。")) return;
  setStatus("清除所有遮罩中...");
  try {
    const summary = await api("/api/clear-all-masks", {});
    applySummary(summary);
    if (app.maskCanvas) {
      app.maskCanvas.getContext("2d").clearRect(0, 0, app.maskCanvas.width, app.maskCanvas.height);
      app.showMaskOverlay = false;
      updateMaskTint();
    }
    renderCanvas();
    updateButtons();
    setStatus(`已清除 ${summary.clearedMasks} 張圖片的遮罩`);
  } catch (error) {
    setStatus(error.message);
  }
}

export async function clearAllDetections() {
  if (!app.images.length) return;
  if (!window.confirm("要清除所有圖片的偵測框嗎？遮罩會保留。")) return;
  setStatus("清除所有偵測框中...");
  try {
    const summary = await api("/api/clear-all-detections", {});
    applySummary(summary);
    renderCanvas();
    setStatus(`已清除 ${summary.clearedDetections} 個偵測框`);
  } catch (error) {
    setStatus(error.message);
  }
}

export async function resetCurrentImage() {
  const image = app.images[app.currentIndex];
  if (!imageHasResettableState(image)) return;
  if (!window.confirm("要還原目前圖片嗎？會清除偵測框、遮罩與去水印結果，原始圖片會保留。")) return;
  setStatus("還原目前圖片中...");
  try {
    const updated = await api("/api/reset-image", { index: app.currentIndex });
    app.images[app.currentIndex] = updated;
    await loadCurrentImage(app.currentIndex, false);
    app.showMaskOverlay = false;
    renderCanvas();
    setStatus(`已還原目前圖片：${updated.name}`);
  } catch (error) {
    setStatus(error.message);
  }
}

export async function resetAllImages() {
  if (!app.images.length) return;
  const count = app.images.filter(imageHasResettableState).length;
  if (!count) return;
  if (!window.confirm(`要還原 ${count} 張圖片嗎？會清除偵測框、遮罩與去水印結果，原始圖片會保留。`)) return;
  setStatus("還原所有圖片中...");
  try {
    const summary = await api("/api/reset-all-images", {});
    applySummary(summary);
    if (app.images.length) {
      await loadCurrentImage(Math.min(app.currentIndex, app.images.length - 1), false);
      app.showMaskOverlay = false;
      renderCanvas();
    }
    setStatus(`已還原 ${summary.resetImages} 張圖片`);
  } catch (error) {
    setStatus(error.message);
  }
}

export async function detectCurrent() {
  await startOrAddCurrentOperation("detect");
}

export async function detectAll() {
  if (!app.images.length) return;
  const count = app.images.length;
  const kind = "批量偵測";
  if (!window.confirm(`要對全部 ${count} 張圖片執行偵測嗎？這會更新每張圖片的偵測框與自動遮罩。`)) return;
  app.operationStopRequested = false;
  setStatus(`${displayProgressKind(kind, count)}中：共 ${count} 張...`);
  beginBatchProgress(kind, count, false);
  try {
    const started = await api("/api/start-batch-detect", {
      ...detectionRequestPayload(),
    });
    const job = started.job || {};
    if (!job.id) throw new Error(`${displayProgressKind(kind, count)}沒有取得工作編號`);
    app.activeBatchJobId = job.id;
    if (app.operationStopRequested) {
      await cancelServerProcessing({ jobId: job.id });
      throw new Error("作業已中斷");
    }
    updateTrackedBatchProgress(kind, job);
    const summary = await waitForBatchJob(kind, job.id, { showMaskOverlay: true });
    if (!summary) throw new Error(`${displayProgressKind(kind)}沒有回傳結果`);
    applySummary(summary);
    await loadCurrentImage(Math.min(app.currentIndex, app.images.length - 1), false);
    app.showMaskOverlay = app.images[app.currentIndex]?.hasMask && !app.images[app.currentIndex]?.hasResult;
    renderCanvas();
    const batch = summary.batch || {};
    const failed = Number(batch.failedImages || 0);
    const processed = Number(batch.processedImages || 0);
    const detections = Number(batch.detectionCount || 0);
    const message = `${processed} 張，${detections} 個區域${failed ? `，${failed} 張失敗` : ""}`;
    finishBatchProgress(kind, message, failed > 0);
    setStatus(`${displayProgressKind(kind)}完成：${message}`);
  } catch (error) {
    if (app.operationStopRequested || error.message === "作業已中斷") {
      cancelBatchProgress(kind);
      setStatus(`已中斷${displayProgressKind(kind)}`);
    } else {
      failBatchProgress(kind, error.message);
      setStatus(error.message);
    }
  }
}

export async function processCurrent() {
  await startOrAddCurrentOperation("process");
}

export async function detectAndProcessCurrent() {
  await startOrAddCurrentOperation("detectProcess");
}

export async function processAll() {
  if (!app.images.length) return;
  const count = app.images.filter((image) => image.hasMask).length;
  if (!count) return;
  const kind = "批量去水印";
  if (!window.confirm(`要對 ${count} 張有遮罩的圖片執行去水印嗎？沒有遮罩的圖片會略過。`)) return;
  app.operationStopRequested = false;
  setStatus(`${displayProgressKind(kind, count)}中：共 ${count} 張...`);
  beginBatchProgress(kind, count, false);
  try {
    const started = await api("/api/start-batch-process", {
      ...processRequestPayload(),
    });
    const job = started.job || {};
    if (!job.id) throw new Error(`${displayProgressKind(kind, count)}沒有取得工作編號`);
    app.activeBatchJobId = job.id;
    if (app.operationStopRequested) {
      await cancelServerProcessing({ jobId: job.id });
      throw new Error("作業已中斷");
    }
    updateTrackedBatchProgress(kind, job);
    const summary = await waitForBatchJob(kind, job.id);
    if (!summary) throw new Error(`${displayProgressKind(kind)}沒有回傳結果`);
    applySummary(summary);
    await loadCurrentImage(Math.min(app.currentIndex, app.images.length - 1), false);
    app.showMaskOverlay = false;
    renderCanvas();
    const batch = summary.batch || {};
    const failed = Number(batch.failedImages || 0);
    const processed = Number(batch.processedImages || 0);
    const message = `${processed} 張${failed ? `，${failed} 張失敗` : ""}`;
    finishBatchProgress(kind, message, failed > 0);
    setStatus(`${displayProgressKind(kind)}完成：${message}`);
  } catch (error) {
    if (app.operationStopRequested || error.message === "作業已中斷") {
      cancelBatchProgress(kind);
      setStatus(`已中斷${displayProgressKind(kind)}`);
    } else {
      failBatchProgress(kind, error.message);
      setStatus(error.message);
    }
  }
}

export function startDownload(path) {
  const link = document.createElement("a");
  link.href = downloadUrl(path);
  document.body.appendChild(link);
  link.click();
  link.remove();
}

export function downloadCurrentImage() {
  const image = app.images[app.currentIndex];
  if (!image?.hasResult) return;
  startDownload(`/api/download-image?index=${image.index}`);
  setStatus(`已開始下載目前圖片：${image.name}`);
}

export function downloadAllImages() {
  const resultCount = app.images.filter((image) => image.hasResult).length;
  if (!resultCount) return;
  startDownload("/api/download-all-images");
  setStatus(`已開始下載全部圖片壓縮包：${resultCount} 張`);
}

export async function detectAndProcessAll() {
  if (!app.images.length) return;
  const count = app.images.length;
  const detectKind = "批量偵測";
  const processKind = "批量去水印";
  const combinedKind = "批量一鍵去水印";
  if (!window.confirm(`要對全部 ${count} 張圖片先偵測再去水印嗎？這會更新每張圖片的偵測框、遮罩和輸出結果。`)) return;
  app.operationStopRequested = false;
  setStatus(`${displayProgressKind(combinedKind, count)}中：共 ${count} 張...`);
  const combinedStartedAt = Date.now();
  beginBatchProgress(detectKind, count, false);
  try {
    const detectedStarted = await api("/api/start-batch-detect", {
      ...detectionRequestPayload(),
    });
    const detectJob = detectedStarted.job || {};
    if (!detectJob.id) throw new Error(`${displayProgressKind(detectKind, count)}沒有取得工作編號`);
    app.activeBatchJobId = detectJob.id;
    if (app.operationStopRequested) {
      await cancelServerProcessing({ jobId: detectJob.id });
      throw new Error("作業已中斷");
    }
    updateTrackedBatchProgress(detectKind, detectJob);
    const detectSummary = await waitForBatchJob(detectKind, detectJob.id, { showMaskOverlay: true });
    if (!detectSummary) throw new Error(`${displayProgressKind(detectKind)}沒有回傳結果`);
    if (app.operationStopRequested) throw new Error("作業已中斷");
    applySummary(detectSummary);
    const detectBatch = detectSummary.batch || {};
    const detected = Number(detectBatch.detectionCount || 0);
    const detectFailed = Number(detectBatch.failedImages || 0);
    if (detectFailed > 0 && detected <= 0) {
      await loadCurrentImage(Math.min(app.currentIndex, app.images.length - 1), false);
      app.showMaskOverlay = false;
      renderCanvas();
      const message = `偵測有 ${detectFailed} 張失敗，未執行去水印`;
      finishBatchProgress(combinedKind, message, true);
      setStatus(`${displayProgressKind(detectKind)}失敗：${message}`);
      return;
    }
    if (detected <= 0) {
      await loadCurrentImage(Math.min(app.currentIndex, app.images.length - 1), false);
      app.showMaskOverlay = false;
      renderCanvas();
      finishBatchProgress(combinedKind, "沒有找到水印，未執行去水印", false);
      setStatus(`${displayProgressKind(detectKind)}完成：沒有找到水印，未執行去水印`);
      return;
    }

    if (app.operationStopRequested) throw new Error("作業已中斷");
    const processCount = app.images.filter((image) => image.hasMask).length;
    if (processCount <= 0) {
      await loadCurrentImage(Math.min(app.currentIndex, app.images.length - 1), false);
      app.showMaskOverlay = false;
      renderCanvas();
      finishBatchProgress(combinedKind, "沒有可去水印的遮罩，未執行去水印", false);
      setStatus(`${displayProgressKind(detectKind)}完成：沒有可去水印的遮罩，未執行去水印`);
      return;
    }
    beginBatchProgress(processKind, processCount, false);
    app.batchProgressStartedAt = combinedStartedAt;
    const processedStarted = await api("/api/start-batch-process", {
      ...processRequestPayload(),
    });
    const processJob = processedStarted.job || {};
    if (!processJob.id) throw new Error(`${displayProgressKind(processKind, processCount)}沒有取得工作編號`);
    app.activeBatchJobId = processJob.id;
    if (app.operationStopRequested) {
      await cancelServerProcessing({ jobId: processJob.id });
      throw new Error("作業已中斷");
    }
    updateTrackedBatchProgress(processKind, processJob);
    const processSummary = await waitForBatchJob(processKind, processJob.id);
    if (!processSummary) throw new Error(`${displayProgressKind(processKind)}沒有回傳結果`);
    applySummary(processSummary);
    await loadCurrentImage(Math.min(app.currentIndex, app.images.length - 1), false);
    app.showMaskOverlay = false;
    renderCanvas();

    const processBatch = processSummary.batch || {};
    const processed = Number(processBatch.processedImages || 0);
    const failed = detectFailed + Number(processBatch.failedImages || 0);
    const message = `${processed} 張，${detected} 個區域${failed ? `，${failed} 張失敗` : ""}`;
    finishBatchProgress(combinedKind, message, failed > 0);
    setStatus(`${displayProgressKind(combinedKind)}完成：${message}`);
  } catch (error) {
    if (app.operationStopRequested || error.message === "作業已中斷") {
      cancelBatchProgress(combinedKind);
      setStatus(`已中斷${displayProgressKind(combinedKind)}`);
    } else {
      failBatchProgress(combinedKind, error.message);
      setStatus(error.message);
    }
  }
}

export async function persistRestoredResult() {
  if (!app.images.length || !app.resultCanvas) return;
  setStatus("還原筆刷已套用，儲存中...");
  try {
    const updated = await api("/api/save", {
      index: app.currentIndex,
      resultData: app.resultCanvas.toDataURL("image/png"),
      maskData: app.maskCanvas.toDataURL("image/png"),
    });
    app.images[app.currentIndex] = updated;
    renderImageList();
    setStatus(`還原修正已暫存，可用下載按鈕取回：${updated.name}`);
  } catch (error) {
    setStatus(error.message);
  }
}

export async function createTemplateFromMask() {
  const image = app.images[app.currentIndex];
  if (!image?.hasMask || !app.maskCanvas) return;
  const suggestedName = `${image.name.replace(/\.[^.]+$/, "")}_template`;
  const name = window.prompt("模板名稱", suggestedName);
  if (name === null) return;
  setStatus("建立模板中...");
  try {
    const summary = await api("/api/create-template", {
      index: app.currentIndex,
      maskData: app.maskCanvas.toDataURL("image/png"),
      name: name.trim() || suggestedName,
    });
    app.images = summary.images;
    app.templates = summary.templates;
    if (summary.createdTemplate?.path) {
      app.selectedTemplates.add(summary.createdTemplate.path);
    }
    renderImageList();
    renderTemplateList();
    renderDetectionList();
    setStatus(`已建立模板：${summary.createdTemplate.name}`);
  } catch (error) {
    setStatus(error.message);
  }
}

export function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error(`無法讀取圖片：${file.name}`));
    reader.readAsDataURL(file);
  });
}

export function supportedImageFiles(files) {
  return Array.from(files || []).filter((file) => {
    const name = file.name || "";
    const extension = name.includes(".") ? name.slice(name.lastIndexOf(".")).toLowerCase() : "";
    return file.type.startsWith("image/") || IMAGE_FILE_EXTENSIONS.has(extension);
  });
}

export function ignoredFileCount(files, supportedFiles) {
  return Math.max(0, Array.from(files || []).length - supportedFiles.length);
}

export function ignoredSuffix(count) {
  return count > 0 ? `，略過 ${count} 個非圖片檔` : "";
}

export async function addImagesFromFiles(files) {
  const selectedFiles = supportedImageFiles(files);
  const ignored = ignoredFileCount(files, selectedFiles);
  if (!selectedFiles.length) {
    setStatus("沒有可匯入的圖片");
    return;
  }
  setStatus(`新增 ${selectedFiles.length} 張圖片中...`);
  try {
    const uploads = await Promise.all(selectedFiles.map(async (file) => ({
      name: file.name,
      data: await fileToDataUrl(file),
    })));
    const summary = await api("/api/add-images", { files: uploads });
    const addedPaths = new Set(summary.addedImages || []);
    const wasProcessing = app.busy;
    applySummary(summary);
    const firstAddedIndex = app.images.findIndex((image) => addedPaths.has(image.path));
    if (wasProcessing) {
      await loadCurrentOrEmpty(app.currentIndex);
      setStatus(`已新增 ${summary.addedImages.length} 張圖片${ignoredSuffix(ignored)}，未加入目前處理`);
    } else {
      await loadCurrentOrEmpty(firstAddedIndex >= 0 ? firstAddedIndex : app.currentIndex);
      setStatus(`已新增 ${summary.addedImages.length} 張圖片${ignoredSuffix(ignored)}`);
    }
  } catch (error) {
    setStatus(error.message);
  }
}

export async function addTemplatesFromFiles(files) {
  const selectedFiles = supportedImageFiles(files);
  const ignored = ignoredFileCount(files, selectedFiles);
  if (!selectedFiles.length) {
    setStatus("沒有可匯入的模板圖片");
    return;
  }
  setStatus(`匯入 ${selectedFiles.length} 張模板中...`);
  try {
    const uploads = await Promise.all(selectedFiles.map(async (file) => ({
      name: file.name,
      data: await fileToDataUrl(file),
    })));
    const summary = await api("/api/add-templates", { files: uploads });
    const addedPaths = new Set(summary.addedTemplates || []);
    applySummary(summary);
    addedPaths.forEach((path) => app.selectedTemplates.add(path));
    renderTemplateList();
    setStatus(`已匯入 ${summary.addedTemplates.length} 張模板${ignoredSuffix(ignored)}`);
  } catch (error) {
    setStatus(error.message);
  }
}

export async function deleteTemplate(template) {
  if (!window.confirm(`要刪除模板「${template.name}」嗎？檔案會移到輸出資料夾的 .editor_trash。`)) return;
  setStatus("刪除模板中...");
  try {
    const summary = await api("/api/delete-template", { template: template.path });
    app.selectedTemplates.delete(template.path);
    applySummary(summary);
    if (app.selectedTemplates.size === 0) {
      app.templates.forEach((item) => app.selectedTemplates.add(item.path));
      renderTemplateList();
    }
    setStatus(`已刪除模板：${template.name}`);
  } catch (error) {
    setStatus(error.message);
  }
}

export async function deleteCurrentImage() {
  if (!app.images.length) return;
  const image = app.images[app.currentIndex];
  if (!window.confirm(`要刪除目前圖片「${image.name}」嗎？檔案會移到輸出資料夾的 .editor_trash。`)) return;
  const nextIndex = app.currentIndex;
  setStatus("刪除圖片中...");
  try {
    const summary = await api("/api/delete-image", { index: app.currentIndex });
    applySummary(summary);
    await loadCurrentOrEmpty(Math.min(nextIndex, app.images.length - 1), "尚未載入圖片");
    setStatus(`已刪除圖片：${image.name}`);
  } catch (error) {
    setStatus(error.message);
  }
}

export async function deleteAllImages() {
  if (!app.images.length) return;
  if (!window.confirm(`要刪除全部 ${app.images.length} 張圖片嗎？檔案會移到輸出資料夾的 .editor_trash。`)) return;
  setStatus("刪除所有圖片中...");
  try {
    const summary = await api("/api/delete-all-images", {});
    applySummary(summary);
    await loadCurrentOrEmpty(0, "尚未載入圖片");
    setStatus(`已刪除 ${summary.deletedImages} 張圖片`);
  } catch (error) {
    setStatus(error.message);
  }
}

export async function configureWorkspace() {
  const templateValue = document.getElementById("templatePath").value;
  const templates = templateValue.split(";").map((item) => item.trim()).filter(Boolean);
  setStatus("載入工作區...");
  try {
    app.selectedTemplates.clear();
    await api("/api/workspace", {
      input: document.getElementById("inputPath").value,
      templates,
      output: document.getElementById("outputPath").value,
    });
    app.currentIndex = 0;
    await refreshState();
    setStatus("工作區已載入");
  } catch (error) {
    setStatus(error.message);
  }
}

setRenderActions({
  deleteTemplate,
  hasOperationBatchItem,
  loadCurrentImage,
});

