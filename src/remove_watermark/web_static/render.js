import {
  BASE_MAX_ZOOM,
  DEFAULT_AI_SETTINGS,
  DEFAULT_TEMPLATE_SETTINGS,
  app,
  emptyState,
} from "./state.js";
import { templateThumbnailUrl, thumbnailUrl } from "./api.js";
import { renderCanvas, updateCursorCoordinates } from "./canvas.js";
import {
  loadSavedAdvancedSettings,
  normalizeAiSettings,
  normalizeTemplateSettings,
  syncAiSettingControls,
  syncTemplateSettingControls,
} from "./settings.js";

let renderActions = {
  deleteTemplate: () => {},
  hasOperationBatchItem: () => false,
  loadCurrentImage: () => {},
  openTemplatePreviewModal: () => {},
  syncTemplatePreviewWithTemplateList: () => {},
};

export function setRenderActions(actions) {
  renderActions = { ...renderActions, ...actions };
}

export function imageIdentity(image) {
  return image?.imageKey || image?.path || "";
}

export function statusLabel(status) {
  return {
    pending: "未處理",
    detected: "已偵測",
    edited: "已編輯",
    processed: "已去除",
    saved: "已儲存",
    error: "失敗",
    failed: "失敗",
  }[status] || status || "未處理";
}

export function syncWorkspaceFields(summary) {
  if ("input" in summary || "inputIsTemporary" in summary) {
    document.getElementById("inputPath").value = summary.inputIsTemporary ? "" : (summary.input || "");
  }
  if (Array.isArray(summary.templateRoots)) {
    document.getElementById("templatePath").value = summary.templateRoots.join(";");
  }
  if ("output" in summary) {
    document.getElementById("outputPath").value = summary.output || "";
  }
  if ("aiPrompt" in summary) {
    const aiPromptInput = document.getElementById("aiPrompt");
    if (!aiPromptInput.value.trim()) {
      app.aiPrompt = summary.aiPrompt || app.aiPrompt;
      aiPromptInput.value = app.aiPrompt;
    }
  }
  if ("savedAdvancedSettings" in summary) {
    app.savedAdvancedSettings = summary.savedAdvancedSettings || {};
  }
  if (summary.aiSettings && !app.aiSettingsInitialized) {
    app.defaultAiSettings = normalizeAiSettings(summary.aiSettings, DEFAULT_AI_SETTINGS);
    const saved = loadSavedAdvancedSettings();
    app.aiSettings = saved.aiSettings
      ? normalizeAiSettings(saved.aiSettings, app.defaultAiSettings)
      : { ...app.defaultAiSettings };
    app.aiSettingsInitialized = true;
    syncAiSettingControls();
  }
  if (summary.templateSettings && !app.templateSettingsInitialized) {
    app.defaultTemplateSettings = normalizeTemplateSettings(summary.templateSettings, DEFAULT_TEMPLATE_SETTINGS);
    const saved = loadSavedAdvancedSettings();
    app.templateSettings = saved.templateSettings
      ? normalizeTemplateSettings(saved.templateSettings, app.defaultTemplateSettings)
      : { ...app.defaultTemplateSettings };
    app.templateSettingsInitialized = true;
    syncTemplateSettingControls();
  }
}

export function applySummary(summary) {
  const currentKey = imageIdentity(app.images[app.currentIndex]);
  app.images = summary.images;
  app.templates = summary.templates;
  renderActions.syncTemplatePreviewWithTemplateList();
  syncWorkspaceFields(summary);
  const preservedIndex = currentKey ? app.images.findIndex((image) => imageIdentity(image) === currentKey) : -1;
  if (preservedIndex >= 0) {
    app.currentIndex = preservedIndex;
  } else if (app.currentIndex >= app.images.length) {
    app.currentIndex = Math.max(0, app.images.length - 1);
  }
  renderImageList();
  renderTemplateList();
  renderDetectionList();
  updateButtons();
}

export function clearCurrentImageState(message = "尚未載入圖片") {
  app.currentIndex = 0;
  app.originalCanvas = null;
  app.resultCanvas = null;
  app.maskCanvas = null;
  app.maskTintCanvas = null;
  app.loadedImageIndex = -1;
  app.showMaskOverlay = false;
  app.pointer = null;
  app.maxZoom = BASE_MAX_ZOOM;
  app.zoom = 1;
  document.getElementById("zoomRange").max = String(BASE_MAX_ZOOM * 100);
  document.getElementById("zoomRange").value = "100";
  emptyState.textContent = message;
  emptyState.style.display = "grid";
  document.getElementById("imageMeta").textContent = "";
  updateCursorCoordinates();
  renderImageList();
  renderDetectionList();
  updateButtons();
  renderCanvas();
}

export function renderImageList() {
  const list = document.getElementById("imageList");
  list.innerHTML = "";
  const currentNumber = app.images.length ? app.currentIndex + 1 : 0;
  document.getElementById("imageCount").textContent = `${currentNumber} / ${app.images.length}`;
  app.images.forEach((image) => {
    const button = document.createElement("button");
    button.className = `image-row${image.index === app.currentIndex ? " active" : ""}`;
    button.type = "button";
    button.title = image.name;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", image.index === app.currentIndex ? "true" : "false");

    const thumbWrap = document.createElement("span");
    thumbWrap.className = "image-thumb-wrap";

    const thumb = document.createElement("img");
    thumb.className = "image-thumb";
    thumb.src = thumbnailUrl(image);
    thumb.alt = image.name;
    thumb.loading = "lazy";
    thumb.draggable = false;

    const index = document.createElement("span");
    index.className = "image-index";
    index.textContent = String(image.index + 1);

    const info = document.createElement("span");
    const name = document.createElement("span");
    name.className = "image-name";
    name.textContent = image.name;
    const status = document.createElement("span");
    status.className = "image-status";
    status.textContent = statusLabel(image.status);

    thumbWrap.append(thumb, index);
    info.append(name, status);
    button.append(thumbWrap, info);
    button.addEventListener("click", () => renderActions.loadCurrentImage(image.index, true));
    list.appendChild(button);
  });
  const active = list.querySelector(".image-row.active");
  if (active) {
    requestAnimationFrame(() => active.scrollIntoView({ block: "nearest", inline: "nearest" }));
  }
}

export function renderTemplateList() {
  const list = document.getElementById("templateList");
  list.innerHTML = "";
  if (!app.templates.length) {
    list.innerHTML = `<div class="template-item template-empty">沒有可用模板</div>`;
    return;
  }
  app.templates.forEach((template) => {
    const item = document.createElement("div");
    item.className = "template-item";
    const checked = app.selectedTemplates.has(template.path);
    const thumbWrap = document.createElement("button");
    thumbWrap.className = "template-thumb-wrap";
    thumbWrap.type = "button";
    thumbWrap.title = `放大查看：${template.name}`;
    thumbWrap.setAttribute("aria-label", `放大查看：${template.name}`);
    const thumb = document.createElement("img");
    thumb.className = "template-thumb";
    thumb.src = templateThumbnailUrl(template);
    thumb.alt = template.name;
    thumb.loading = "lazy";
    thumb.draggable = false;
    thumbWrap.append(thumb);

    const details = document.createElement("div");
    details.className = "template-details";
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = template.path;
    checkbox.checked = checked;
    const name = document.createElement("strong");
    name.textContent = template.name;
    name.title = template.name;
    const path = document.createElement("span");
    path.className = "template-path";
    path.textContent = template.path;
    path.title = template.path;
    const deleteButton = document.createElement("button");
    deleteButton.className = "template-delete small ghost has-icon";
    deleteButton.type = "button";
    deleteButton.title = `刪除模板：${template.name}`;
    deleteButton.setAttribute("aria-label", `刪除模板：${template.name}`);
    deleteButton.innerHTML = `<svg class="button-icon" aria-hidden="true"><use href="#icon-trash"></use></svg>`;

    checkbox.addEventListener("change", () => {
      if (checkbox.checked) app.selectedTemplates.add(template.path);
      else app.selectedTemplates.delete(template.path);
      updateButtons();
    });
    thumbWrap.addEventListener("click", () => renderActions.openTemplatePreviewModal(template));
    deleteButton.addEventListener("click", () => renderActions.deleteTemplate(template));

    label.append(checkbox, name);
    details.append(label, path);
    item.append(thumbWrap, details, deleteButton);
    list.append(item);
  });
}

export function renderDetectionList() {
  const image = app.images[app.currentIndex];
  const detections = image?.detections || [];
  const detectionCount = document.getElementById("detectionCount");
  detectionCount.textContent = String(detections.length);
  detectionCount.setAttribute("aria-label", `偵測數量：${detections.length}`);
  const list = document.getElementById("detectionList");
  list.innerHTML = "";
  if (!detections.length) {
    list.innerHTML = `<div class="detection-item">尚無偵測結果</div>`;
    return;
  }
  detections.forEach((detection, index) => {
    const bbox = detection.bbox || [0, 0, 0, 0];
    const rawScore = detection.score;
    const score = rawScore === null || rawScore === "" || rawScore === undefined ? NaN : Number(rawScore);
    const hasScore = Number.isFinite(score);
    const threshold = Number(image?.templateSettings?.scoreThreshold);
    const reviewKept = image?.detector === "template" && hasScore && Number.isFinite(threshold) && score < threshold;
    const item = document.createElement("div");
    item.className = "detection-item";
    if (reviewKept) item.classList.add("review-kept");

    const title = document.createElement("strong");
    title.textContent = `#${index + 1} ${detection.template || ""}`;
    title.title = title.textContent;

    const meta = document.createElement("span");
    meta.className = "detection-meta";
    const scoreText = document.createElement("span");
    scoreText.textContent = `分數 ${hasScore ? score.toFixed(3) : "-"}，位置 ${bbox.join(", ")}`;
    meta.appendChild(scoreText);

    if (reviewKept) {
      const badge = document.createElement("span");
      badge.className = "detection-badge";
      badge.title = "分數低於候選初篩，但其他複檢條件通過；判斷使用未四捨五入分數";
      badge.textContent = "複檢保留";
      meta.appendChild(badge);
    }

    item.append(title, meta);
    const diagnostics = renderDetectionDiagnostics(detection.diagnostics);
    if (diagnostics) item.appendChild(diagnostics);
    list.appendChild(item);
  });
}

function renderDetectionDiagnostics(diagnostics) {
  if (!diagnostics || typeof diagnostics !== "object") return null;

  const details = document.createElement("details");
  details.className = "detection-diagnostics";
  const summary = document.createElement("summary");
  summary.textContent = "詳細";
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "detection-diagnostics-grid";
  appendDiagnosticRow(body, "Profile", diagnostics.profile || "-");
  appendDiagnosticRow(body, "最終門檻", formatDiagnosticNumber(diagnostics.acceptanceThreshold));
  appendDiagnosticRow(body, "Fit 門檻", formatDiagnosticNumber(diagnostics.fitThreshold));
  appendDiagnosticRow(body, "風險扣分", formatDiagnosticNumber(diagnostics.risk));
  appendDiagnosticRow(body, "證據數", diagnostics.evidenceCount ?? "-");
  appendDiagnosticRow(body, "有效群組", formatDiagnosticList(diagnostics.activeGroups));

  const scoreGroups = diagnostics.scoreGroups && typeof diagnostics.scoreGroups === "object"
    ? diagnostics.scoreGroups
    : {};
  Object.keys(scoreGroups).sort().forEach((name) => {
    appendDiagnosticRow(body, `group:${name}`, formatDiagnosticNumber(scoreGroups[name]));
  });

  details.appendChild(body);
  return details;
}

function appendDiagnosticRow(parent, label, value) {
  const key = document.createElement("span");
  key.className = "detection-diagnostics-key";
  key.textContent = label;
  const val = document.createElement("span");
  val.className = "detection-diagnostics-value";
  val.textContent = String(value);
  parent.append(key, val);
}

function formatDiagnosticNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "-";
}

function formatDiagnosticList(value) {
  return Array.isArray(value) && value.length ? value.join(", ") : "-";
}

export function imageHasResettableState(image) {
  return Boolean(
    image
      && (
        image.hasMask
        || image.hasResult
        || image.error
        || image.status !== "pending"
        || (image.detections || []).length > 0
      )
  );
}

export function updateButtons() {
  const currentImage = app.images[app.currentIndex];
  const detections = app.images[app.currentIndex]?.detections || [];
  const hasAnyMask = app.images.some((image) => image.hasMask);
  const hasAnyDetections = app.images.some((image) => (image.detections || []).length > 0);
  const hasAnyResult = app.images.some((image) => image.hasResult);
  const hasAnyResettableState = app.images.some(imageHasResettableState);
  const operationRunning = Boolean(app.operationMode);
  const blockingBusy = app.busy && !operationRunning;
  const needsTemplate = app.detectorMode === "template";
  const canDetect = app.images.length > 0 && !(needsTemplate && app.selectedTemplates.size === 0);
  const canProcess = app.images.length > 0 && Boolean(currentImage?.hasMask) && Boolean(app.maskCanvas) && app.loadedImageIndex === app.currentIndex;
  const canDetectProcess = canDetect;
  const canAddMode = (mode, baseAllowed) => {
    if (!baseAllowed) return false;
    if (!operationRunning) return !app.busy;
    if (app.operationStopRequested) return false;
    return Boolean(app.activeBatchJobId) && app.operationMode === mode && !renderActions.hasOperationBatchItem(app.currentIndex, mode);
  };
  document.getElementById("loadWorkspace").disabled = app.busy;
  document.getElementById("prevImage").disabled = blockingBusy || app.currentIndex <= 0;
  document.getElementById("nextImage").disabled = blockingBusy || app.currentIndex >= app.images.length - 1;
  document.getElementById("addImage").disabled = false;
  document.getElementById("addImageFolder").disabled = false;
  document.getElementById("addTemplate").disabled = app.busy;
  document.getElementById("addTemplateFolder").disabled = app.busy;
  document.getElementById("createTemplate").disabled = app.busy || app.images.length === 0 || !currentImage?.hasMask || !app.maskCanvas;
  document.getElementById("deleteImage").disabled = app.busy || app.images.length === 0;
  document.getElementById("deleteAllImages").disabled = app.busy || app.images.length === 0;
  document.getElementById("detectImage").disabled = !canAddMode("detect", canDetect);
  document.getElementById("processImage").disabled = !canAddMode("process", canProcess);
  document.getElementById("detectProcessImage").disabled = !canAddMode("detectProcess", canDetectProcess);
  document.getElementById("downloadImage").disabled = app.busy || !currentImage?.hasResult;
  document.getElementById("batchDetect").disabled = app.busy || app.images.length === 0 || (needsTemplate && app.selectedTemplates.size === 0);
  document.getElementById("batchProcess").disabled = app.busy || app.images.length === 0 || !hasAnyMask;
  document.getElementById("batchDetectProcess").disabled = app.busy || app.images.length === 0 || (needsTemplate && app.selectedTemplates.size === 0);
  document.getElementById("cancelProcessing").disabled = !app.busy;
  document.getElementById("downloadAllImages").disabled = app.busy || !hasAnyResult;
  document.getElementById("clearMask").disabled = app.busy || !app.maskCanvas;
  document.getElementById("clearDetections").disabled = app.busy || detections.length === 0;
  document.getElementById("clearAllMasks").disabled = app.busy || !hasAnyMask;
  document.getElementById("clearAllDetections").disabled = app.busy || !hasAnyDetections;
  document.getElementById("resetImage").disabled = app.busy || !imageHasResettableState(currentImage);
  document.getElementById("resetAllImages").disabled = app.busy || !hasAnyResettableState;
  document.getElementById("keepDetectionsAfterProcess").disabled = app.busy;
}
