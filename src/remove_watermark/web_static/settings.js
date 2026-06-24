import {
  ADVANCED_SETTINGS_STORAGE_KEY,
  DEFAULT_AI_SETTINGS,
  DEFAULT_TEMPLATE_SETTINGS,
  app,
} from "./state.js";
import { api } from "./api.js";

let settingsActions = {
  setStatus: () => {},
  refreshTemplatePreview: () => {},
  updateButtons: () => {},
};

export function configureSettingsActions(actions) {
  settingsActions = { ...settingsActions, ...actions };
}

export function setDetectorMode(mode) {
  app.detectorMode = ["template", "sam3"].includes(mode) ? mode : "template";
  document.querySelectorAll(".mode-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.detector === app.detectorMode);
  });
  const usesAiSettings = app.detectorMode === "sam3";
  const aiDetectMode = app.detectorMode === "sam3";
  document.getElementById("aiSettingsPanel").hidden = !usesAiSettings;
  document.getElementById("aiPromptRow").hidden = app.detectorMode !== "sam3";
  document.getElementById("templateSettingsPanel").hidden = app.detectorMode !== "template";
  document.getElementById("templatePanel").hidden = app.detectorMode !== "template";
  document.getElementById("aiBoxThresholdLabel").textContent = "信心門檻";
  document.getElementById("aiSettingsTitle").textContent = "SAM3 進階參數";
  document.getElementById("aiSettingsHint").textContent = "模型門檻、遮罩、分塊";
  document.querySelectorAll("[data-ai-scope='detect']").forEach((row) => {
    row.hidden = !aiDetectMode;
  });
  settingsActions.updateButtons();
}

export function syncAiSettingControls() {
  document.getElementById("sam3ConfidenceThreshold").value = formatAiDecimal(app.aiSettings.sam3ConfidenceThreshold);
  document.getElementById("aiBoxThreshold").value = formatAiDecimal(app.aiSettings.boxThreshold);
  document.getElementById("aiMaxBoxAreaRatio").value = formatAiDecimal(app.aiSettings.maxBoxAreaRatio);
  document.getElementById("aiNmsIouThreshold").value = formatAiDecimal(app.aiSettings.nmsIouThreshold);
  document.getElementById("aiMaxDetections").value = String(app.aiSettings.maxDetections);
  document.getElementById("aiMaskThreshold").value = formatAiDecimal(app.aiSettings.maskThreshold);
  document.getElementById("aiMaskDilatePixels").value = String(app.aiSettings.maskDilatePixels);
  document.getElementById("aiFallbackToBoxes").checked = Boolean(app.aiSettings.fallbackToBoxes);
  document.getElementById("sam3MaxSide").value = String(app.aiSettings.sam3MaxSide);
  document.getElementById("sam3TileOverlapRatio").value = formatAiDecimal(app.aiSettings.sam3TileOverlapRatio);
}

export function syncTemplateSettingControls() {
  document.getElementById("templateScoreThreshold").value = formatAiDecimal(app.templateSettings.scoreThreshold);
  document.getElementById("templateMinScale").value = formatAiDecimal(app.templateSettings.minScale);
  document.getElementById("templateMaxScale").value = formatAiDecimal(app.templateSettings.maxScale);
  document.getElementById("templateScaleStep").value = formatAiDecimal(app.templateSettings.scaleStep);
  document.getElementById("templateMaxDetections").value = String(app.templateSettings.maxDetections);
  document.getElementById("templateNmsIouThreshold").value = formatAiDecimal(app.templateSettings.nmsIouThreshold);
  document.getElementById("templateEdgeScoreThreshold").value = formatAiDecimal(app.templateSettings.edgeScoreThreshold);
  document.getElementById("templateColorScoreThreshold").value = formatAiDecimal(app.templateSettings.colorScoreThreshold);
  document.getElementById("templateSupportCorrelationThreshold").value = formatAiDecimal(app.templateSettings.supportCorrelationThreshold);
  document.getElementById("templateMaskDilateIterations").value = String(app.templateSettings.maskDilateIterations);
  document.getElementById("templateMaskDilateMaxBodyRatio").value = formatAiDecimal(app.templateSettings.maskDilateMaxBodyRatio);
  document.getElementById("templateMaskEdgeFeatherPixels").value = formatAiDecimal(app.templateSettings.maskEdgeFeatherPixels);
  document.getElementById("templateMaskUnifyBody").checked = Boolean(app.templateSettings.maskUnifyBody);
  document.getElementById("templateSam3RefineMask").checked = Boolean(app.templateSettings.sam3RefineMask);
  document.getElementById("templateMaskContourClosePixels").value = formatAiDecimal(app.templateSettings.maskContourClosePixels);
  document.getElementById("templateMaskBodyGapRatio").value = formatAiDecimal(app.templateSettings.maskBodyGapRatio);
  syncTemplateBodyMaskControls();
}

export function formatAiDecimal(value) {
  return Number(value).toFixed(2);
}

export function clampNumber(value, min, max, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

export function booleanSetting(value, fallback) {
  return typeof value === "boolean" ? value : fallback;
}

export function normalizeAiSettings(settings = {}, defaults = DEFAULT_AI_SETTINGS) {
  const source = settings && typeof settings === "object" ? settings : {};
  return {
    sam3ConfidenceThreshold: clampNumber(source.sam3ConfidenceThreshold, 0, 1, defaults.sam3ConfidenceThreshold),
    boxThreshold: clampNumber(source.boxThreshold, 0.01, 1, defaults.boxThreshold),
    maxBoxAreaRatio: clampNumber(source.maxBoxAreaRatio, 0.01, 1, defaults.maxBoxAreaRatio),
    nmsIouThreshold: clampNumber(source.nmsIouThreshold, 0.01, 1, defaults.nmsIouThreshold),
    maxDetections: Math.round(clampNumber(source.maxDetections, 1, 96, defaults.maxDetections)),
    maskThreshold: clampNumber(source.maskThreshold, 0, 1, defaults.maskThreshold),
    maskDilatePixels: Math.round(clampNumber(source.maskDilatePixels, 0, 64, defaults.maskDilatePixels)),
    fallbackToBoxes: booleanSetting(source.fallbackToBoxes, defaults.fallbackToBoxes),
    sam3MaxSide: Math.round(clampNumber(source.sam3MaxSide, 256, 8192, defaults.sam3MaxSide)),
    sam3TileOverlapRatio: clampNumber(source.sam3TileOverlapRatio, 0, 0.8, defaults.sam3TileOverlapRatio),
  };
}

export function normalizeTemplateSettings(settings = {}, defaults = DEFAULT_TEMPLATE_SETTINGS) {
  const source = settings && typeof settings === "object" ? settings : {};
  const minScale = clampNumber(source.minScale, 0.05, 5, defaults.minScale);
  const maxScale = clampNumber(source.maxScale, 0.05, 5, defaults.maxScale);
  return {
    scoreThreshold: clampNumber(source.scoreThreshold, 0.05, 1, defaults.scoreThreshold),
    minScale: Math.min(minScale, maxScale),
    maxScale: Math.max(minScale, maxScale),
    scaleStep: clampNumber(source.scaleStep, 0.01, 1, defaults.scaleStep),
    maxDetections: Math.round(clampNumber(source.maxDetections, 1, 240, defaults.maxDetections)),
    nmsIouThreshold: clampNumber(source.nmsIouThreshold, 0.01, 1, defaults.nmsIouThreshold),
    edgeScoreThreshold: clampNumber(source.edgeScoreThreshold, 0.01, 1, defaults.edgeScoreThreshold),
    colorScoreThreshold: clampNumber(source.colorScoreThreshold, 0.01, 1, defaults.colorScoreThreshold),
    supportCorrelationThreshold: clampNumber(source.supportCorrelationThreshold, 0.01, 1, defaults.supportCorrelationThreshold),
    maskDilateIterations: Math.round(clampNumber(source.maskDilateIterations, 0, 64, defaults.maskDilateIterations)),
    maskDilateMaxBodyRatio: clampNumber(source.maskDilateMaxBodyRatio, 0, 1, defaults.maskDilateMaxBodyRatio),
    maskEdgeFeatherPixels: clampNumber(source.maskEdgeFeatherPixels, 0, 16, defaults.maskEdgeFeatherPixels),
    maskUnifyBody: booleanSetting(source.maskUnifyBody, defaults.maskUnifyBody),
    sam3RefineMask: booleanSetting(source.sam3RefineMask, defaults.sam3RefineMask),
    maskContourClosePixels: clampNumber(source.maskContourClosePixels, 0, 32, defaults.maskContourClosePixels),
    maskBodyGapRatio: clampNumber(source.maskBodyGapRatio, 0, 0.5, defaults.maskBodyGapRatio),
  };
}

export function loadSavedAdvancedSettings() {
  if (hasSavedAdvancedSettings(app.savedAdvancedSettings)) {
    return normalizeSavedAdvancedSettings(app.savedAdvancedSettings);
  }
  try {
    const raw = localStorage.getItem(ADVANCED_SETTINGS_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    return normalizeSavedAdvancedSettings(parsed);
  } catch {
    return {};
  }
}

function hasSavedAdvancedSettings(settings) {
  return Boolean(settings?.aiSettings || settings?.templateSettings);
}

export function normalizeSavedAdvancedSettings(settings) {
  const next = {};
  if (settings?.aiSettings) {
    next.aiSettings = normalizeAiSettings(settings.aiSettings, app.defaultAiSettings);
  }
  if (settings?.templateSettings) {
    next.templateSettings = normalizeTemplateSettings(settings.templateSettings, app.defaultTemplateSettings);
  }
  return next;
}

function writeLegacySavedAdvancedSettings(settings) {
  if (hasSavedAdvancedSettings(settings)) {
    localStorage.setItem(ADVANCED_SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } else {
    localStorage.removeItem(ADVANCED_SETTINGS_STORAGE_KEY);
  }
}

export async function writeSavedAdvancedSettings(settings) {
  const next = normalizeSavedAdvancedSettings(settings);
  const summary = await api("/api/settings", next);
  app.savedAdvancedSettings = normalizeSavedAdvancedSettings(summary.savedAdvancedSettings || next);
  try {
    writeLegacySavedAdvancedSettings(app.savedAdvancedSettings);
  } catch {
    // 後端設定檔已保存；瀏覽器舊快取失敗不影響使用。
  }
  return summary;
}

export async function saveAiSettings() {
  const saved = loadSavedAdvancedSettings();
  saved.aiSettings = readAiSettings();
  try {
    await writeSavedAdvancedSettings(saved);
    settingsActions.setStatus("已保存 SAM3 進階設定");
  } catch {
    settingsActions.setStatus("保存 SAM3 進階設定失敗");
  }
}

export async function resetAiSettings() {
  app.aiSettings = { ...app.defaultAiSettings };
  syncAiSettingControls();
  const saved = loadSavedAdvancedSettings();
  delete saved.aiSettings;
  try {
    await writeSavedAdvancedSettings(saved);
    settingsActions.setStatus("已重置 SAM3 進階設定");
  } catch {
    settingsActions.setStatus("已重置畫面，但清除 SAM3 保存設定失敗");
  }
}

export async function saveTemplateSettings() {
  const saved = loadSavedAdvancedSettings();
  saved.templateSettings = readTemplateSettings();
  settingsActions.refreshTemplatePreview();
  try {
    await writeSavedAdvancedSettings(saved);
    settingsActions.setStatus("已保存模板進階設定");
  } catch {
    settingsActions.setStatus("保存模板進階設定失敗");
  }
}

export async function resetTemplateSettings() {
  app.templateSettings = { ...app.defaultTemplateSettings };
  syncTemplateSettingControls();
  settingsActions.refreshTemplatePreview();
  const saved = loadSavedAdvancedSettings();
  delete saved.templateSettings;
  try {
    await writeSavedAdvancedSettings(saved);
    settingsActions.setStatus("已重置模板進階設定");
  } catch {
    settingsActions.setStatus("已重置畫面，但清除模板保存設定失敗");
  }
}

export function readAiSettings() {
  app.aiSettings = {
    sam3ConfidenceThreshold: clampNumber(document.getElementById("sam3ConfidenceThreshold").value, 0, 1, app.aiSettings.sam3ConfidenceThreshold),
    boxThreshold: clampNumber(document.getElementById("aiBoxThreshold").value, 0.01, 1, app.aiSettings.boxThreshold),
    maxBoxAreaRatio: clampNumber(document.getElementById("aiMaxBoxAreaRatio").value, 0.01, 1, app.aiSettings.maxBoxAreaRatio),
    nmsIouThreshold: clampNumber(document.getElementById("aiNmsIouThreshold").value, 0.01, 1, app.aiSettings.nmsIouThreshold),
    maxDetections: Math.round(clampNumber(document.getElementById("aiMaxDetections").value, 1, 96, app.aiSettings.maxDetections)),
    maskThreshold: clampNumber(document.getElementById("aiMaskThreshold").value, 0, 1, app.aiSettings.maskThreshold),
    maskDilatePixels: Math.round(clampNumber(document.getElementById("aiMaskDilatePixels").value, 0, 64, app.aiSettings.maskDilatePixels)),
    fallbackToBoxes: document.getElementById("aiFallbackToBoxes").checked,
    sam3MaxSide: Math.round(clampNumber(document.getElementById("sam3MaxSide").value, 256, 8192, app.aiSettings.sam3MaxSide)),
    sam3TileOverlapRatio: clampNumber(document.getElementById("sam3TileOverlapRatio").value, 0, 0.8, app.aiSettings.sam3TileOverlapRatio),
  };
  syncAiSettingControls();
  return app.aiSettings;
}

export function activeAiSettings() {
  return readAiSettings();
}

export function readTemplateSettings() {
  const minScale = clampNumber(document.getElementById("templateMinScale").value, 0.05, 5, app.templateSettings.minScale);
  const maxScale = clampNumber(document.getElementById("templateMaxScale").value, 0.05, 5, app.templateSettings.maxScale);
  app.templateSettings = {
    scoreThreshold: clampNumber(document.getElementById("templateScoreThreshold").value, 0.05, 1, app.templateSettings.scoreThreshold),
    minScale: Math.min(minScale, maxScale),
    maxScale: Math.max(minScale, maxScale),
    scaleStep: clampNumber(document.getElementById("templateScaleStep").value, 0.01, 1, app.templateSettings.scaleStep),
    maxDetections: Math.round(clampNumber(document.getElementById("templateMaxDetections").value, 1, 240, app.templateSettings.maxDetections)),
    nmsIouThreshold: clampNumber(document.getElementById("templateNmsIouThreshold").value, 0.01, 1, app.templateSettings.nmsIouThreshold),
    edgeScoreThreshold: clampNumber(document.getElementById("templateEdgeScoreThreshold").value, 0.01, 1, app.templateSettings.edgeScoreThreshold),
    colorScoreThreshold: clampNumber(document.getElementById("templateColorScoreThreshold").value, 0.01, 1, app.templateSettings.colorScoreThreshold),
    supportCorrelationThreshold: clampNumber(
      document.getElementById("templateSupportCorrelationThreshold").value,
      0.01,
      1,
      app.templateSettings.supportCorrelationThreshold,
    ),
    maskDilateIterations: Math.round(clampNumber(
      document.getElementById("templateMaskDilateIterations").value,
      0,
      64,
      app.templateSettings.maskDilateIterations,
    )),
    maskDilateMaxBodyRatio: clampNumber(
      document.getElementById("templateMaskDilateMaxBodyRatio").value,
      0,
      1,
      app.templateSettings.maskDilateMaxBodyRatio,
    ),
    maskEdgeFeatherPixels: clampNumber(
      document.getElementById("templateMaskEdgeFeatherPixels").value,
      0,
      16,
      app.templateSettings.maskEdgeFeatherPixels,
    ),
    maskUnifyBody: document.getElementById("templateMaskUnifyBody").checked,
    sam3RefineMask: document.getElementById("templateSam3RefineMask").checked,
    maskContourClosePixels: clampNumber(
      document.getElementById("templateMaskContourClosePixels").value,
      0,
      32,
      app.templateSettings.maskContourClosePixels,
    ),
    maskBodyGapRatio: clampNumber(
      document.getElementById("templateMaskBodyGapRatio").value,
      0,
      0.5,
      app.templateSettings.maskBodyGapRatio,
    ),
  };
  syncTemplateSettingControls();
  return app.templateSettings;
}

export function syncTemplateBodyMaskControls() {
  const disabled = !Boolean(app.templateSettings.maskUnifyBody);
  document.getElementById("templateMaskContourClosePixels").disabled = disabled;
  document.getElementById("templateMaskBodyGapRatio").disabled = disabled;
}

export function detectionRequestPayload() {
  if (app.detectorMode === "sam3") {
    app.aiPrompt = document.getElementById("aiPrompt").value.trim() || app.aiPrompt;
  }
  return {
    detector: app.detectorMode,
    aiPrompt: app.detectorMode === "sam3" ? app.aiPrompt : "",
    aiSettings: app.detectorMode === "sam3" ? activeAiSettings() : null,
    templateSettings: app.detectorMode === "template" ? readTemplateSettings() : null,
    templates: app.detectorMode === "template" ? Array.from(app.selectedTemplates) : [],
  };
}

export function processRequestPayload() {
  app.keepDetectionsAfterProcess = document.getElementById("keepDetectionsAfterProcess").checked;
  return {
    keepDetectionsAfterProcess: app.keepDetectionsAfterProcess,
  };
}
