export const ADVANCED_SETTINGS_STORAGE_KEY = "removeWatermark.advancedSettings.v1";

export const DEFAULT_AI_SETTINGS = {
  sam3ConfidenceThreshold: 0.05,
  boxThreshold: 0.20,
  maxBoxAreaRatio: 0.35,
  nmsIouThreshold: 0.30,
  maxDetections: 60,
  maskThreshold: 0.00,
  maskDilatePixels: 3,
  fallbackToBoxes: true,
  sam3MaxSide: 2048,
  sam3TileOverlapRatio: 0.20,
};

export const DEFAULT_TEMPLATE_SETTINGS = {
  scoreThreshold: 0.50,
  minScale: 1.00,
  maxScale: 1.00,
  scaleStep: 0.05,
  maxDetections: 96,
  nmsIouThreshold: 0.30,
  edgeScoreThreshold: 0.30,
  colorScoreThreshold: 0.55,
  supportCorrelationThreshold: 0.12,
  maskDilateIterations: 10,
  maskDilateMaxBodyRatio: 0.10,
  maskEdgeFeatherPixels: 2.00,
  maskUnifyBody: true,
  maskContourClosePixels: 3.00,
  maskBodyGapRatio: 0.05,
  sam3RefineMask: false,
};

export const app = {
  images: [],
  templates: [],
  currentIndex: 0,
  selectedTemplates: new Set(),
  templatePreviewTemplate: null,
  templatePreviewMode: "original",
  templatePreviewZoom: 1,
  templatePreviewReturnFocus: null,
  detectorMode: "sam3",
  keepDetectionsAfterProcess: false,
  aiPrompt: "watermark. text watermark. transparent watermark. logo. stamp.",
  aiSettings: { ...DEFAULT_AI_SETTINGS },
  defaultAiSettings: { ...DEFAULT_AI_SETTINGS },
  aiSettingsInitialized: false,
  templateSettings: { ...DEFAULT_TEMPLATE_SETTINGS },
  defaultTemplateSettings: { ...DEFAULT_TEMPLATE_SETTINGS },
  templateSettingsInitialized: false,
  savedAdvancedSettings: {},
  mode: "sideBySide",
  tool: "pan",
  showMaskOverlay: false,
  zoom: 1,
  maxZoom: 30,
  pan: { x: 0, y: 0 },
  compareSplit: 0.5,
  brushSize: 36,
  drawing: false,
  panning: false,
  draggingCompareSplit: false,
  leftButtonDown: false,
  loadRequestId: 0,
  lastPoint: null,
  pointer: null,
  originalCanvas: null,
  resultCanvas: null,
  maskCanvas: null,
  maskTintCanvas: null,
  loadedImageIndex: -1,
  canvas: document.getElementById("editorCanvas"),
  brushCursor: document.getElementById("brushCursor"),
  ctx: null,
  busy: false,
  operationMode: "",
  operationStopRequested: false,
  operationPaths: new Set(),
  activeBatchJobId: "",
  batchProgressTimer: null,
  batchProgressStartedAt: 0,
  batchProgressTotal: 0,
  batchProgressKind: "",
  batchProgressHideTimer: null,
  batchProgressLatest: null,
  batchProgressAppliedKey: "",
  renderFrame: 0,
  gridPatterns: new Map(),
};

export const statusText = document.getElementById("statusText");
export const processingTime = document.getElementById("processingTime");
export const cursorCoordinates = document.getElementById("cursorCoordinates");
export const emptyState = document.getElementById("emptyState");
export const batchProgress = document.getElementById("batchProgress");
export const batchProgressLabel = document.getElementById("batchProgressLabel");
export const batchProgressFill = document.getElementById("batchProgressFill");
export const batchProgressMeta = document.getElementById("batchProgressMeta");
export const MIN_ZOOM = 0.15;
export const BASE_MAX_ZOOM = 30;
export const TARGET_MAX_PIXEL_SCALE = 2;
export const ABSOLUTE_MAX_ZOOM = 500;
export const IMAGE_FILE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]);
export const GRID_SPACING = 14;
export const GRID_MAJOR_INTERVAL = 5;
export const GRID_DOT_RADIUS = 0.7;
export const GRID_MAJOR_DOT_RADIUS = 1.15;
export const GRID_DOT_COLOR = "rgba(177, 190, 181, 0.25)";
export const GRID_MAJOR_DOT_COLOR = "rgba(94, 196, 167, 0.16)";
export const CANVAS_GRID_BASE = "#0c0f0e";
export const PANEL_GRID_BASE = "#101411";
export const COMPARE_SLIDER_HIT_RADIUS = 20;
