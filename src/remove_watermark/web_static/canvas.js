import {
  ABSOLUTE_MAX_ZOOM,
  BASE_MAX_ZOOM,
  CANVAS_GRID_BASE,
  COMPARE_SLIDER_HIT_RADIUS,
  GRID_DOT_COLOR,
  GRID_DOT_RADIUS,
  GRID_MAJOR_DOT_COLOR,
  GRID_MAJOR_DOT_RADIUS,
  GRID_MAJOR_INTERVAL,
  GRID_SPACING,
  MIN_ZOOM,
  PANEL_GRID_BASE,
  TARGET_MAX_PIXEL_SCALE,
  app,
  cursorCoordinates,
} from "./state.js";

export function setCursorCoordinates(hit = null) {
  if (!cursorCoordinates) return;
  if (!hit || !app.originalCanvas) {
    cursorCoordinates.textContent = "座標 --";
    return;
  }
  const x = Math.min(app.originalCanvas.width - 1, Math.max(0, Math.floor(hit.x)));
  const y = Math.min(app.originalCanvas.height - 1, Math.max(0, Math.floor(hit.y)));
  cursorCoordinates.textContent = `座標 x ${x}, y ${y}`;
}

export function updateCursorCoordinates(point = app.pointer) {
  setCursorCoordinates(point ? screenToImage(point) : null);
}

export function makeSurface(width, height) {
  const surface = document.createElement("canvas");
  surface.width = width;
  surface.height = height;
  return surface;
}

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.loading = "eager";
    image.onload = () => {
      const decoded = typeof image.decode === "function" ? image.decode().catch(() => undefined) : Promise.resolve();
      decoded.then(() => resolve(image));
    };
    image.onerror = () => reject(new Error(`無法載入影像：${src}`));
    image.src = src;
  });
}

export function drawImageToSurface(image, width, height, contextOptions) {
  const surface = makeSurface(width, height);
  const ctx = surface.getContext("2d", contextOptions);
  ctx.drawImage(image, 0, 0, width, height);
  return surface;
}

export function buildMaskSurface(image, width, height) {
  const source = drawImageToSurface(image, width, height, { willReadFrequently: true });
  const srcCtx = source.getContext("2d", { willReadFrequently: true });
  const pixels = srcCtx.getImageData(0, 0, width, height);
  for (let i = 0; i < pixels.data.length; i += 4) {
    const value = Math.max(pixels.data[i], pixels.data[i + 1], pixels.data[i + 2]);
    pixels.data[i] = 255;
    pixels.data[i + 1] = 255;
    pixels.data[i + 2] = 255;
    pixels.data[i + 3] = value;
  }
  const mask = makeSurface(width, height);
  mask.getContext("2d", { willReadFrequently: true }).putImageData(pixels, 0, 0);
  return mask;
}

export function updateMaskTint() {
  if (!app.maskCanvas) return;
  app.maskTintCanvas = makeSurface(app.maskCanvas.width, app.maskCanvas.height);
  const ctx = app.maskTintCanvas.getContext("2d");
  ctx.clearRect(0, 0, app.maskTintCanvas.width, app.maskTintCanvas.height);
  ctx.drawImage(app.maskCanvas, 0, 0);
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = "rgba(94, 196, 167, 0.68)";
  ctx.fillRect(0, 0, app.maskTintCanvas.width, app.maskTintCanvas.height);
  ctx.globalCompositeOperation = "source-over";
}

export function resizeCanvas() {
  const rect = app.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  app.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  app.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  app.ctx = app.canvas.getContext("2d");
  app.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  app.gridPatterns.clear();
  updateZoomLimit();
  renderCanvas();
}

export function canvasSize() {
  const rect = app.canvas.getBoundingClientRect();
  return { width: rect.width, height: rect.height };
}

export function panels() {
  const size = canvasSize();
  const gap = 12;
  if (app.mode === "sideBySide") {
    const width = (size.width - gap) / 2;
    return [
      { kind: "original", label: "原圖", x: 0, y: 0, w: width, h: size.height },
      { kind: "result", label: "處理結果", x: width + gap, y: 0, w: width, h: size.height },
    ];
  }
  if (app.mode === "compare") {
    return [{ kind: "compare", label: "對比", x: 0, y: 0, w: size.width, h: size.height }];
  }
  const kind = app.mode === "original" ? "original" : "result";
  return [{ kind, label: app.modeLabel || "", x: 0, y: 0, w: size.width, h: size.height }];
}

export function fitScaleForPanel(panel) {
  const imageWidth = app.originalCanvas.width;
  const imageHeight = app.originalCanvas.height;
  const availableWidth = Math.max(1, panel.w - 34);
  const availableHeight = Math.max(1, panel.h - 34);
  return Math.min(availableWidth / imageWidth, availableHeight / imageHeight);
}

export function dynamicMaxZoom() {
  if (!app.originalCanvas) return BASE_MAX_ZOOM;
  const fit = panels().reduce((smallest, panel) => Math.min(smallest, fitScaleForPanel(panel)), Infinity);
  if (!Number.isFinite(fit) || fit <= 0) return BASE_MAX_ZOOM;
  const zoomForPixelDetail = TARGET_MAX_PIXEL_SCALE / fit;
  return Math.min(ABSOLUTE_MAX_ZOOM, Math.max(BASE_MAX_ZOOM, Math.ceil(zoomForPixelDetail)));
}

export function clampZoom(value) {
  return Math.min(app.maxZoom || BASE_MAX_ZOOM, Math.max(MIN_ZOOM, value));
}

export function setBrushSize(value) {
  const brushRange = document.getElementById("brushSize");
  const min = Number(brushRange.min || 4);
  const max = Number(brushRange.max || 160);
  app.brushSize = Math.min(max, Math.max(min, Math.round(value)));
  brushRange.value = String(app.brushSize);
  document.getElementById("brushSizeLabel").textContent = String(app.brushSize);
  updateBrushCursor();
}

export function updateZoomLimit() {
  app.maxZoom = dynamicMaxZoom();
  app.zoom = clampZoom(app.zoom);
  const zoomRange = document.getElementById("zoomRange");
  const maxPercent = Math.round(app.maxZoom * 100);
  zoomRange.max = String(maxPercent);
  zoomRange.value = String(Math.round(app.zoom * 100));
  zoomRange.title = `最大 ${maxPercent}%`;
}

export function transformFor(panel) {
  return transformForZoom(panel, app.zoom);
}

export function transformForZoom(panel, zoom) {
  const fit = fitScaleForPanel(panel);
  const scale = Math.max(0.02, fit * zoom);
  return {
    scale,
    x: panel.x + (panel.w - app.originalCanvas.width * scale) / 2 + app.pan.x,
    y: panel.y + (panel.h - app.originalCanvas.height * scale) / 2 + app.pan.y,
  };
}

export function drawSurface(ctx, surface, transform) {
  ctx.drawImage(surface, transform.x, transform.y, surface.width * transform.scale, surface.height * transform.scale);
}

export function imageBounds(transform) {
  return {
    x: transform.x,
    y: transform.y,
    w: app.originalCanvas.width * transform.scale,
    h: app.originalCanvas.height * transform.scale,
  };
}

export function compareSplitX(transform) {
  const bounds = imageBounds(transform);
  return bounds.x + bounds.w * app.compareSplit;
}

export function gridPattern(ctx, baseColor) {
  const tileSize = GRID_SPACING * GRID_MAJOR_INTERVAL;
  const key = `${baseColor}:${tileSize}`;
  const cached = app.gridPatterns.get(key);
  if (cached) return cached;

  const tile = makeSurface(tileSize, tileSize);
  const tileCtx = tile.getContext("2d");
  tileCtx.fillStyle = baseColor;
  tileCtx.fillRect(0, 0, tileSize, tileSize);

  tileCtx.fillStyle = GRID_DOT_COLOR;
  tileCtx.beginPath();
  for (let py = 0; py < tileSize; py += GRID_SPACING) {
    for (let px = 0; px < tileSize; px += GRID_SPACING) {
      tileCtx.moveTo(px + GRID_DOT_RADIUS, py);
      tileCtx.arc(px, py, GRID_DOT_RADIUS, 0, Math.PI * 2);
    }
  }
  tileCtx.fill();

  tileCtx.fillStyle = GRID_MAJOR_DOT_COLOR;
  tileCtx.beginPath();
  tileCtx.moveTo(GRID_MAJOR_DOT_RADIUS, 0);
  tileCtx.arc(0, 0, GRID_MAJOR_DOT_RADIUS, 0, Math.PI * 2);
  tileCtx.fill();

  const pattern = ctx.createPattern(tile, "repeat");
  if (pattern) app.gridPatterns.set(key, pattern);
  return pattern;
}

export function drawDottedGrid(ctx, x, y, width, height, baseColor) {
  const pattern = gridPattern(ctx, baseColor);
  ctx.fillStyle = pattern || baseColor;
  ctx.fillRect(x, y, width, height);
}

export function drawComparisonPanel(ctx, panel, transform) {
  const bounds = imageBounds(transform);
  const splitX = compareSplitX(transform);

  drawSurface(ctx, app.originalCanvas, transform);

  ctx.save();
  ctx.beginPath();
  ctx.rect(splitX, bounds.y, Math.max(0, bounds.x + bounds.w - splitX), bounds.h);
  ctx.clip();
  drawSurface(ctx, app.resultCanvas, transform);
  if (shouldShowMaskOverlay(panel)) {
    drawSurface(ctx, app.maskTintCanvas, transform);
  }
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = "rgba(240, 179, 91, 0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(splitX, bounds.y);
  ctx.lineTo(splitX, bounds.y + bounds.h);
  ctx.stroke();
  ctx.fillStyle = "#f0b35b";
  ctx.strokeStyle = "rgba(16, 18, 17, 0.9)";
  ctx.lineWidth = 2;
  const handleY = bounds.y + bounds.h / 2;
  ctx.beginPath();
  ctx.arc(splitX, handleY, 13, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#17110a";
  ctx.font = "700 14px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("↔", splitX, handleY - 1);
  ctx.restore();

  drawComparisonLabels(ctx, panel, splitX);
}

export function drawComparisonLabels(ctx, panel, splitX) {
  ctx.save();
  ctx.font = "12px Segoe UI, sans-serif";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(16, 18, 17, 0.78)";
  ctx.fillRect(panel.x + 12, panel.y + 12, 62, 24);
  ctx.fillRect(panel.x + panel.w - 74, panel.y + 12, 62, 24);
  ctx.fillStyle = "#dce4dc";
  ctx.fillText("原圖", panel.x + 22, panel.y + 24);
  ctx.fillText("結果", panel.x + panel.w - 64, panel.y + 24);
  ctx.restore();
}

export function drawDetections(ctx, transform) {
  const detections = app.images[app.currentIndex]?.detections || [];
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#f0b35b";
  ctx.fillStyle = "rgba(16, 18, 17, 0.82)";
  ctx.font = "12px Segoe UI, sans-serif";
  detections.forEach((detection, index) => {
    const [x, y, w, h] = detection.bbox || [0, 0, 0, 0];
    const sx = transform.x + x * transform.scale;
    const sy = transform.y + y * transform.scale;
    const sw = w * transform.scale;
    const sh = h * transform.scale;
    ctx.strokeRect(sx, sy, sw, sh);
    const label = `${index + 1} ${detection.template || ""}`;
    const labelWidth = Math.max(48, ctx.measureText(label).width + 10);
    ctx.fillRect(sx, Math.max(0, sy - 19), labelWidth, 18);
    ctx.fillStyle = "#f4d49b";
    ctx.fillText(label, sx + 5, Math.max(13, sy - 6));
    ctx.fillStyle = "rgba(16, 18, 17, 0.82)";
  });
  ctx.restore();
}

export function renderCanvas() {
  if (app.renderFrame) return;
  app.renderFrame = requestAnimationFrame(() => {
    app.renderFrame = 0;
    renderCanvasNow();
  });
}

export function renderCanvasNow() {
  if (!app.ctx) return;
  const size = canvasSize();
  const ctx = app.ctx;
  ctx.save();
  ctx.clearRect(0, 0, size.width, size.height);
  drawDottedGrid(ctx, 0, 0, size.width, size.height, CANVAS_GRID_BASE);
  if (!app.originalCanvas) {
    ctx.restore();
    updateBrushCursor();
    updateCursorCoordinates();
    return;
  }
  panels().forEach((panel) => {
    const transform = transformFor(panel);
    ctx.save();
    ctx.beginPath();
    ctx.rect(panel.x, panel.y, panel.w, panel.h);
    ctx.clip();
    drawDottedGrid(ctx, panel.x, panel.y, panel.w, panel.h, PANEL_GRID_BASE);
    if (panel.kind === "compare") {
      drawComparisonPanel(ctx, panel, transform);
    } else {
      const surface = panel.kind === "original" ? app.originalCanvas : app.resultCanvas;
      drawSurface(ctx, surface, transform);
      if (shouldShowMaskOverlay(panel)) {
        drawSurface(ctx, app.maskTintCanvas, transform);
      }
    }
    if (app.mode === "sideBySide" || app.mode === "compare") {
      drawDetections(ctx, transform);
    }
    if (panel.kind !== "compare") {
      ctx.fillStyle = "rgba(16, 18, 17, 0.78)";
      ctx.fillRect(panel.x + 12, panel.y + 12, 78, 24);
      ctx.fillStyle = "#dce4dc";
      ctx.font = "12px Segoe UI, sans-serif";
      ctx.fillText(panel.label, panel.x + 22, panel.y + 28);
    }
    ctx.restore();
  });
  ctx.restore();
  document.getElementById("zoomLabel").textContent = `${Math.round(app.zoom * 100)}%`;
  document.getElementById("zoomRangeLabel").textContent = `${Math.round(app.zoom * 100)}%`;
  updateBrushCursor();
  updateCursorCoordinates();
}

export function shouldShowMaskOverlay(panel) {
  if (!app.showMaskOverlay && app.tool !== "brush" && app.tool !== "eraser") return false;
  return app.mode === "result" || panel.kind === "result" || panel.kind === "compare";
}

export function brushCursorHit() {
  if (!app.pointer || app.tool === "pan") return null;
  if (!app.originalCanvas) return { transform: { scale: 1 } };
  const hit = screenToPanelImagePoint(app.pointer) || previewPanelImagePoint();
  if (!hit) return null;
  return hit;
}

export function updateBrushCursor() {
  if (!app.brushCursor) return;
  const hit = brushCursorHit();
  if (!hit) {
    app.brushCursor.hidden = true;
    return;
  }
  const size = Math.max(2, app.brushSize);
  app.brushCursor.hidden = false;
  app.brushCursor.classList.toggle("restore", app.tool === "restore");
  app.brushCursor.style.width = `${size}px`;
  app.brushCursor.style.height = `${size}px`;
  app.brushCursor.style.transform = `translate3d(${app.pointer.x}px, ${app.pointer.y}px, 0) translate(-50%, -50%)`;
}

export function previewPanelImagePoint() {
  const panel = panels()[0];
  if (!panel) return null;
  const transform = transformFor(panel);
  return { x: 0, y: 0, panel, transform };
}

export function eventPoint(event) {
  const rect = app.canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

export function screenToPanelImagePoint(point) {
  if (!app.originalCanvas) return null;
  for (const panel of panels()) {
    if (point.x < panel.x || point.x > panel.x + panel.w || point.y < panel.y || point.y > panel.y + panel.h) continue;
    const transform = transformFor(panel);
    const x = (point.x - transform.x) / transform.scale;
    const y = (point.y - transform.y) / transform.scale;
    return { x, y, panel, transform };
  }
  return null;
}

export function screenToImage(point) {
  const hit = screenToPanelImagePoint(point);
  if (!hit) return null;
  if (hit.x >= 0 && hit.y >= 0 && hit.x <= app.originalCanvas.width && hit.y <= app.originalCanvas.height) {
    return hit;
  }
  return null;
}

export function imageBrushSize(transform) {
  const scale = Math.max(0.02, Number(transform?.scale) || 1);
  return app.brushSize / scale;
}

export function compareSliderHit(point) {
  if (app.mode !== "compare" || !app.originalCanvas) return null;
  const panel = panels()[0];
  if (!panel) return null;
  const transform = transformFor(panel);
  const bounds = imageBounds(transform);
  const splitX = compareSplitX(transform);
  const insideY = point.y >= bounds.y && point.y <= bounds.y + bounds.h;
  if (!insideY || Math.abs(point.x - splitX) > COMPARE_SLIDER_HIT_RADIUS) return null;
  return { panel, transform, bounds, splitX };
}

export function updateCompareSplit(point) {
  if (app.mode !== "compare" || !app.originalCanvas) return;
  const panel = panels()[0];
  const transform = transformFor(panel);
  const bounds = imageBounds(transform);
  app.compareSplit = Math.min(1, Math.max(0, (point.x - bounds.x) / bounds.w));
}

export function zoomTowardPoint(point, nextZoom) {
  const hit = screenToPanelImagePoint(point);
  if (!hit) {
    app.zoom = nextZoom;
    return;
  }
  const nextTransform = transformForZoom(hit.panel, nextZoom);
  const imageWidth = app.originalCanvas.width;
  const imageHeight = app.originalCanvas.height;
  const baseX = hit.panel.x + (hit.panel.w - imageWidth * nextTransform.scale) / 2;
  const baseY = hit.panel.y + (hit.panel.h - imageHeight * nextTransform.scale) / 2;
  app.pan.x = point.x - hit.x * nextTransform.scale - baseX;
  app.pan.y = point.y - hit.y * nextTransform.scale - baseY;
  app.zoom = nextZoom;
}

export function drawBrushLine(from, to, brushSize) {
  const ctx = app.maskCanvas.getContext("2d", { willReadFrequently: true });
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = brushSize;
  if (app.tool === "eraser") {
    ctx.globalCompositeOperation = "destination-out";
    ctx.strokeStyle = "rgba(255,255,255,1)";
  } else {
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = "rgba(255,255,255,1)";
  }
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
  ctx.restore();
  updateMaskTint();
}

export function restoreOriginalLine(from, to, brushSize) {
  const resultCtx = app.resultCanvas.getContext("2d", { willReadFrequently: true });
  const distance = Math.hypot(to.x - from.x, to.y - from.y);
  const steps = Math.max(1, Math.ceil(distance / Math.max(Number.EPSILON, brushSize / 3)));
  for (let step = 0; step <= steps; step += 1) {
    const ratio = step / steps;
    const x = from.x + (to.x - from.x) * ratio;
    const y = from.y + (to.y - from.y) * ratio;
    resultCtx.save();
    resultCtx.beginPath();
    resultCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    resultCtx.clip();
    resultCtx.drawImage(app.originalCanvas, 0, 0);
    resultCtx.restore();
  }
}

export function drawAt(point) {
  const hit = screenToImage(point);
  if (!hit) return false;
  const current = { x: hit.x, y: hit.y };
  const previous = app.lastPoint || current;
  const brushSize = imageBrushSize(hit.transform);
  if (app.tool === "restore") {
    restoreOriginalLine(previous, current, brushSize);
  } else if (app.tool === "brush" || app.tool === "eraser") {
    drawBrushLine(previous, current, brushSize);
  }
  app.lastPoint = current;
  renderCanvas();
  return true;
}
