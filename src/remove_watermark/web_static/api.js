import { app } from "./state.js";

export async function api(path, payload, requestOptions = {}) {
  const fetchOptions = payload === undefined
    ? {}
    : {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      };
  if (requestOptions.signal) fetchOptions.signal = requestOptions.signal;
  const response = await fetch(path, fetchOptions);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "操作失敗");
  }
  return data;
}

export function imageVersion(index, kind) {
  const image = app.images[index];
  if (!image) return String(Date.now());
  if (kind === "original") return image.originalVersion || image.path || String(index);
  if (kind === "result") return image.resultVersion || image.originalVersion || `${image.path || index}:result`;
  if (kind === "mask") return image.maskVersion || `${image.originalVersion || image.path || index}:blank-mask`;
  return image.path || String(index);
}

export function imageUrl(index, kind) {
  return `/api/image?index=${index}&kind=${kind}&v=${encodeURIComponent(imageVersion(index, kind))}`;
}

export function downloadUrl(path) {
  return `${path}${path.includes("?") ? "&" : "?"}v=${Date.now()}`;
}

export function thumbnailUrl(image) {
  const sourceKey = encodeURIComponent(image.originalVersion || image.path || image.name || String(image.index));
  return `/api/image?index=${image.index}&kind=thumbnail&v=${sourceKey}`;
}

export function templateThumbnailUrl(template) {
  const sourceKey = encodeURIComponent(template.path || template.name || String(template.index));
  return `/api/template?index=${template.index}&kind=thumbnail&v=${sourceKey}`;
}

export function templateOriginalUrl(template) {
  const sourceKey = encodeURIComponent(template.path || template.name || String(template.index));
  return `/api/template?index=${template.index}&kind=original&v=${sourceKey}`;
}

export function templateMaskPreviewUrl(template, settings = {}, options = {}) {
  const params = new URLSearchParams({
    index: String(template.index),
    kind: options.full ? "mask-preview-full" : "mask-preview",
  });
  [
    "maskDilateIterations",
    "maskDilateMaxBodyRatio",
    "maskEdgeFeatherPixels",
    "maskUnifyBody",
    "maskContourClosePixels",
    "maskBodyGapRatio",
  ].forEach((key) => {
    if (settings[key] !== undefined) params.set(key, String(settings[key]));
  });
  params.set("v", `${template.path || template.name || template.index}:${params.toString()}`);
  return `/api/template?${params.toString()}`;
}
