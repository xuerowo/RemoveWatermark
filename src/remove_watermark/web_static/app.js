import { app } from "./state.js";
import { templateMaskPreviewUrl, templateOriginalUrl } from "./api.js";
import {
  clampZoom,
  compareSliderHit,
  drawAt,
  eventPoint,
  renderCanvas,
  resizeCanvas,
  screenToImage,
  setBrushSize,
  updateBrushCursor,
  updateCompareSplit,
  updateCursorCoordinates,
  updateMaskTint,
  updateZoomLimit,
  zoomTowardPoint,
} from "./canvas.js";
import {
  addImagesFromFiles,
  addTemplatesFromFiles,
  clearAllDetections,
  clearAllMasks,
  clearDetections,
  configureWorkspace,
  createTemplateFromMask,
  deleteAllImages,
  deleteCurrentImage,
  detectAll,
  detectAndProcessAll,
  detectAndProcessCurrent,
  detectCurrent,
  downloadAllImages,
  downloadCurrentImage,
  loadCurrentImage,
  persistMask,
  persistRestoredResult,
  processAll,
  processCurrent,
  refreshState,
  resetAllImages,
  resetCurrentImage,
  setStatus,
  stopAllProcessing,
} from "./operations.js";
import { renderTemplateList, setRenderActions, updateButtons } from "./render.js";
import {
  configureSettingsActions,
  readAiSettings,
  readTemplateSettings,
  resetAiSettings,
  resetTemplateSettings,
  saveAiSettings,
  saveTemplateSettings,
  setDetectorMode,
  syncAiSettingControls,
  syncTemplateSettingControls,
} from "./settings.js";

function refreshTemplatePreview() {
  if (app.templates.length) renderTemplateList();
  if (app.templatePreviewTemplate && app.templatePreviewMode === "mask") {
    syncTemplatePreviewModal();
  }
}

function syncTemplatePreviewWithTemplateList() {
  const currentTemplates = new Map(app.templates.map((template) => [template.path, template]));
  if (app.templatePreviewTemplate) {
    const currentTemplate = currentTemplates.get(app.templatePreviewTemplate.path);
    if (currentTemplate) {
      app.templatePreviewTemplate = currentTemplate;
      syncTemplatePreviewModal();
    } else {
      closeTemplatePreviewModal();
    }
  }
}

function templatePreviewImageUrl(template, mode, { full = false } = {}) {
  return mode === "mask"
    ? templateMaskPreviewUrl(template, app.templateSettings, { full })
    : templateOriginalUrl(template);
}

function templatePreviewModeLabel(mode) {
  return mode === "mask" ? "遮罩預覽" : "原始模板";
}

function openTemplatePreviewModal(template) {
  app.templatePreviewReturnFocus = document.activeElement;
  app.templatePreviewTemplate = template;
  app.templatePreviewMode = "original";
  app.templatePreviewZoom = 1;
  document.getElementById("templatePreviewModal").hidden = false;
  document.body.classList.add("modal-open");
  syncTemplatePreviewModal();
  requestAnimationFrame(() => document.getElementById("templatePreviewClose").focus({ preventScroll: true }));
}

function closeTemplatePreviewModal() {
  const modal = document.getElementById("templatePreviewModal");
  if (modal) modal.hidden = true;
  document.body.classList.remove("modal-open");
  app.templatePreviewTemplate = null;
  const returnFocus = app.templatePreviewReturnFocus;
  app.templatePreviewReturnFocus = null;
  if (returnFocus && typeof returnFocus.focus === "function") {
    returnFocus.focus({ preventScroll: true });
  }
}

function setTemplatePreviewMode(mode) {
  if (!app.templatePreviewTemplate) return;
  app.templatePreviewMode = mode === "original" ? "original" : "mask";
  app.templatePreviewZoom = 1;
  syncTemplatePreviewModal();
}

function setTemplatePreviewZoom(value) {
  app.templatePreviewZoom = Math.min(8, Math.max(0.25, Number(value) || 1));
  updateTemplatePreviewZoom();
}

function syncTemplatePreviewModal() {
  const template = app.templatePreviewTemplate;
  if (!template) return;
  const modal = document.getElementById("templatePreviewModal");
  if (!modal || modal.hidden) return;
  const image = document.getElementById("templatePreviewImage");
  const title = document.getElementById("templatePreviewTitle");
  const mode = document.getElementById("templatePreviewMode");
  title.textContent = template.name;
  mode.textContent = templatePreviewModeLabel(app.templatePreviewMode);
  document.getElementById("templatePreviewOriginal").classList.toggle("active", app.templatePreviewMode === "original");
  document.getElementById("templatePreviewMask").classList.toggle("active", app.templatePreviewMode === "mask");
  image.alt = `${template.name} ${templatePreviewModeLabel(app.templatePreviewMode)}`;
  image.src = templatePreviewImageUrl(template, app.templatePreviewMode, { full: true });
  updateTemplatePreviewZoom();
}

function updateTemplatePreviewZoom() {
  const image = document.getElementById("templatePreviewImage");
  const surface = document.getElementById("templatePreviewSurface");
  const zoomLabel = document.getElementById("templatePreviewZoomLabel");
  if (!image || !surface || !zoomLabel) return;
  zoomLabel.textContent = `${Math.round(app.templatePreviewZoom * 100)}%`;
  if (!image.naturalWidth || !image.naturalHeight) {
    surface.style.width = "";
    surface.style.height = "";
    return;
  }
  surface.style.width = `${Math.max(1, Math.round(image.naturalWidth * app.templatePreviewZoom))}px`;
  surface.style.height = `${Math.max(1, Math.round(image.naturalHeight * app.templatePreviewZoom))}px`;
}

function trapTemplatePreviewFocus(event) {
  if (event.key !== "Tab" || !app.templatePreviewTemplate) return;
  const modal = document.getElementById("templatePreviewModal");
  const focusable = Array.from(modal.querySelectorAll("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])"))
    .filter((element) => element.tabIndex >= 0 && !element.disabled && element.offsetParent !== null);
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function bindEvents() {
  window.addEventListener("resize", resizeCanvas);
  app.canvas.addEventListener("pointerdown", (event) => {
    if (!app.originalCanvas) return;
    if (event.button !== 0) return;
    app.canvas.setPointerCapture(event.pointerId);
    const point = eventPoint(event);
    app.pointer = point;
    updateCursorCoordinates(point);
    app.leftButtonDown = true;
    if (compareSliderHit(point)) {
      app.draggingCompareSplit = true;
      updateCompareSplit(point);
      app.canvas.classList.add("comparing");
      renderCanvas();
    } else if (app.tool === "pan") {
      app.panning = true;
      app.lastPoint = point;
      app.canvas.classList.add("panning");
    } else if (screenToImage(point)) {
      app.drawing = true;
      app.lastPoint = null;
      drawAt(point);
      app.canvas.classList.add("painting");
    } else {
      app.drawing = false;
      app.lastPoint = null;
    }
  });
  app.canvas.addEventListener("pointermove", (event) => {
    const point = eventPoint(event);
    app.pointer = point;
    updateCursorCoordinates(point);
    if (app.draggingCompareSplit) {
      updateCompareSplit(point);
      renderCanvas();
    } else if (app.panning && app.lastPoint) {
      app.pan.x += point.x - app.lastPoint.x;
      app.pan.y += point.y - app.lastPoint.y;
      app.lastPoint = point;
      renderCanvas();
    } else if (app.drawing) {
      drawAt(point);
    } else {
      app.canvas.classList.toggle("comparing", !!compareSliderHit(point));
      updateBrushCursor();
    }
  });
  app.canvas.addEventListener("pointerup", async (event) => {
    app.canvas.releasePointerCapture(event.pointerId);
    const point = eventPoint(event);
    app.pointer = point;
    updateCursorCoordinates(point);
    if (event.button === 0) app.leftButtonDown = false;
    const changedMask = app.drawing && (app.tool === "brush" || app.tool === "eraser");
    const changedResult = app.drawing && app.tool === "restore";
    app.drawing = false;
    app.panning = false;
    app.draggingCompareSplit = false;
    app.lastPoint = null;
    app.canvas.classList.remove("comparing");
    app.canvas.classList.remove("panning");
    app.canvas.classList.toggle("painting", app.tool !== "pan");
    updateButtons();
    if (changedMask) await persistMask();
    if (changedResult) await persistRestoredResult();
  });
  app.canvas.addEventListener("pointerleave", () => {
    if (!app.leftButtonDown) {
      app.pointer = null;
      app.canvas.classList.remove("comparing");
    }
    updateCursorCoordinates();
    updateBrushCursor();
  });
  app.canvas.addEventListener("pointercancel", () => {
    app.pointer = null;
    updateCursorCoordinates();
    app.leftButtonDown = false;
    app.drawing = false;
    app.panning = false;
    app.draggingCompareSplit = false;
    app.lastPoint = null;
    app.canvas.classList.remove("comparing");
    app.canvas.classList.remove("panning");
    app.canvas.classList.toggle("painting", app.tool !== "pan");
    renderCanvas();
  });
  app.canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const point = eventPoint(event);
    app.pointer = point;
    updateCursorCoordinates(point);
    if (app.tool !== "pan") {
      const direction = event.deltaY < 0 ? 1 : -1;
      const step = event.shiftKey ? 1 : 4;
      setBrushSize(app.brushSize + direction * step);
      return;
    }
    const factor = event.deltaY < 0 ? 1.1 : 0.9;
    const nextZoom = clampZoom(app.zoom * factor);
    zoomTowardPoint(point, nextZoom);
    document.getElementById("zoomRange").value = String(Math.round(app.zoom * 100));
    renderCanvas();
  }, { passive: false });

  document.querySelectorAll(".view-tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".view-tab").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      app.mode = button.dataset.view;
      app.draggingCompareSplit = false;
      app.canvas.classList.remove("comparing");
      updateZoomLimit();
      renderCanvas();
    });
  });
  document.querySelectorAll(".tool-button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tool-button").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      app.tool = button.dataset.tool;
      app.canvas.classList.remove("panning");
      app.canvas.classList.toggle("painting", app.tool !== "pan");
      renderCanvas();
    });
  });

  document.getElementById("brushSize").addEventListener("input", (event) => {
    setBrushSize(Number(event.target.value));
  });
  document.getElementById("zoomRange").addEventListener("input", (event) => {
    app.zoom = clampZoom(Number(event.target.value) / 100);
    event.target.value = String(Math.round(app.zoom * 100));
    renderCanvas();
  });
  document.getElementById("clearMask").addEventListener("click", async () => {
    if (!app.maskCanvas) return;
    app.maskCanvas.getContext("2d").clearRect(0, 0, app.maskCanvas.width, app.maskCanvas.height);
    app.showMaskOverlay = false;
    updateMaskTint();
    renderCanvas();
    await persistMask(false);
  });
  document.getElementById("cancelProcessing").addEventListener("click", stopAllProcessing);
  document.getElementById("prevImage").addEventListener("click", () => loadCurrentImage(app.currentIndex - 1, true));
  document.getElementById("nextImage").addEventListener("click", () => loadCurrentImage(app.currentIndex + 1, true));
  document.getElementById("addImage").addEventListener("click", () => document.getElementById("addImageInput").click());
  document.getElementById("addImageInput").addEventListener("change", async (event) => {
    await addImagesFromFiles(event.target.files);
    event.target.value = "";
  });
  document.getElementById("addImageFolder").addEventListener("click", () => document.getElementById("addImageFolderInput").click());
  document.getElementById("addImageFolderInput").addEventListener("change", async (event) => {
    await addImagesFromFiles(event.target.files);
    event.target.value = "";
  });
  document.getElementById("deleteImage").addEventListener("click", deleteCurrentImage);
  document.getElementById("deleteAllImages").addEventListener("click", deleteAllImages);
  document.querySelectorAll(".mode-button").forEach((button) => {
    button.addEventListener("click", () => setDetectorMode(button.dataset.detector));
  });
  document.getElementById("aiPrompt").addEventListener("input", (event) => {
    app.aiPrompt = event.target.value;
  });
  document.getElementById("keepDetectionsAfterProcess").addEventListener("change", (event) => {
    app.keepDetectionsAfterProcess = event.target.checked;
  });
  [
    "sam3ConfidenceThreshold",
    "aiBoxThreshold",
    "aiMaxBoxAreaRatio",
    "aiNmsIouThreshold",
    "aiMaxDetections",
    "aiMaskThreshold",
    "aiMaskDilatePixels",
    "aiFallbackToBoxes",
    "sam3MaxSide",
    "sam3TileOverlapRatio",
  ].forEach((id) => {
    document.getElementById(id).addEventListener("change", readAiSettings);
  });
  [
    "templateScoreThreshold",
    "templateMinScale",
    "templateMaxScale",
    "templateScaleStep",
    "templateMaxDetections",
    "templateNmsIouThreshold",
    "templateEdgeScoreThreshold",
    "templateColorScoreThreshold",
    "templateSupportCorrelationThreshold",
    "templateMaskDilateIterations",
    "templateMaskDilateMaxBodyRatio",
    "templateMaskEdgeFeatherPixels",
    "templateMaskUnifyBody",
    "templateSam3RefineMask",
    "templateMaskContourClosePixels",
    "templateMaskBodyGapRatio",
  ].forEach((id) => {
    document.getElementById(id).addEventListener("change", () => {
      readTemplateSettings();
      refreshTemplatePreview();
    });
  });
  document.getElementById("resetAiSettings").addEventListener("click", resetAiSettings);
  document.getElementById("saveAiSettings").addEventListener("click", saveAiSettings);
  document.getElementById("resetTemplateSettings").addEventListener("click", resetTemplateSettings);
  document.getElementById("saveTemplateSettings").addEventListener("click", saveTemplateSettings);
  document.getElementById("detectImage").addEventListener("click", detectCurrent);
  document.getElementById("processImage").addEventListener("click", processCurrent);
  document.getElementById("detectProcessImage").addEventListener("click", detectAndProcessCurrent);
  document.getElementById("downloadImage").addEventListener("click", downloadCurrentImage);
  document.getElementById("loadWorkspace").addEventListener("click", configureWorkspace);
  document.getElementById("clearDetections").addEventListener("click", clearDetections);
  document.getElementById("clearAllMasks").addEventListener("click", clearAllMasks);
  document.getElementById("clearAllDetections").addEventListener("click", clearAllDetections);
  document.getElementById("resetImage").addEventListener("click", resetCurrentImage);
  document.getElementById("resetAllImages").addEventListener("click", resetAllImages);
  document.getElementById("batchDetect").addEventListener("click", detectAll);
  document.getElementById("batchProcess").addEventListener("click", processAll);
  document.getElementById("batchDetectProcess").addEventListener("click", detectAndProcessAll);
  document.getElementById("downloadAllImages").addEventListener("click", downloadAllImages);
  document.getElementById("addTemplate").addEventListener("click", () => document.getElementById("addTemplateInput").click());
  document.getElementById("addTemplateInput").addEventListener("change", async (event) => {
    await addTemplatesFromFiles(event.target.files);
    event.target.value = "";
  });
  document.getElementById("addTemplateFolder").addEventListener("click", () => document.getElementById("addTemplateFolderInput").click());
  document.getElementById("addTemplateFolderInput").addEventListener("change", async (event) => {
    await addTemplatesFromFiles(event.target.files);
    event.target.value = "";
  });
  document.getElementById("createTemplate").addEventListener("click", createTemplateFromMask);
  document.getElementById("templatePreviewClose").addEventListener("click", closeTemplatePreviewModal);
  document.getElementById("templatePreviewBackdrop").addEventListener("click", closeTemplatePreviewModal);
  document.getElementById("templatePreviewOriginal").addEventListener("click", () => setTemplatePreviewMode("original"));
  document.getElementById("templatePreviewMask").addEventListener("click", () => setTemplatePreviewMode("mask"));
  document.getElementById("templatePreviewZoomOut").addEventListener("click", () => setTemplatePreviewZoom(app.templatePreviewZoom / 1.25));
  document.getElementById("templatePreviewZoomIn").addEventListener("click", () => setTemplatePreviewZoom(app.templatePreviewZoom * 1.25));
  document.getElementById("templatePreviewZoomReset").addEventListener("click", () => setTemplatePreviewZoom(1));
  document.getElementById("templatePreviewImage").addEventListener("load", updateTemplatePreviewZoom);
  document.getElementById("templatePreviewViewport").addEventListener("wheel", (event) => {
    if (!event.ctrlKey) return;
    event.preventDefault();
    setTemplatePreviewZoom(app.templatePreviewZoom * (event.deltaY < 0 ? 1.12 : 1 / 1.12));
  }, { passive: false });
  window.addEventListener("keydown", (event) => {
    if (!app.templatePreviewTemplate) return;
    trapTemplatePreviewFocus(event);
    if (event.defaultPrevented) return;
    if (event.key === "Escape") closeTemplatePreviewModal();
    if (event.key === "+" || event.key === "=") setTemplatePreviewZoom(app.templatePreviewZoom * 1.25);
    if (event.key === "-") setTemplatePreviewZoom(app.templatePreviewZoom / 1.25);
    if (event.key === "0") setTemplatePreviewZoom(1);
  });
}

async function init() {
  app.ctx = app.canvas.getContext("2d");
  bindEvents();
  syncAiSettingControls();
  syncTemplateSettingControls();
  setDetectorMode(app.detectorMode);
  resizeCanvas();
  try {
    await refreshState();
    setStatus("準備就緒");
  } catch (error) {
    setStatus(error.message);
  }
}

configureSettingsActions({
  refreshTemplatePreview,
  setStatus,
  updateButtons,
});
setRenderActions({
  openTemplatePreviewModal,
  syncTemplatePreviewWithTemplateList,
});

init();
