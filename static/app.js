const video = document.querySelector("#camera");
const canvas = document.querySelector("#snapshot");
const emptyState = document.querySelector("#emptyState");
const startBtn = document.querySelector("#startBtn");
const captureBtn = document.querySelector("#captureBtn");
const clearBtn = document.querySelector("#clearBtn");
const uploadInput = document.querySelector("#uploadInput");
const uploadButtonText = document.querySelector("#uploadButtonText");
const dropZone = document.querySelector("#dropZone");
const uploadPrompt = document.querySelector("#uploadPrompt");
const imagePreview = document.querySelector("#imagePreview");
const analysisOverlay = document.querySelector("#analysisOverlay");
const cameraStage = document.querySelector("#cameraStage");
const modelSelect = document.querySelector("#modelSelect");
const emptyResult = document.querySelector("#emptyResult");
const resultContent = document.querySelector("#resultContent");
const resultStatus = document.querySelector("#resultStatus");
const resultPlant = document.querySelector("#resultPlant");
const resultLabel = document.querySelector("#resultLabel");
const resultSummary = document.querySelector("#resultSummary");
const confidenceText = document.querySelector("#confidenceText");
const confidenceBar = document.querySelector("#confidenceBar");
const confidenceTrack = document.querySelector(".confidence-bar");
const topPredictions = document.querySelector("#topPredictions");
const recommendationTitle = document.querySelector("#recommendationTitle");
const recommendationText = document.querySelector("#recommendationText");
const modelName = document.querySelector("#modelName");
const modelVariant = document.querySelector("#modelVariant");

let stream = null;
let previewUrl = null;
let currentBlob = null;
let availableModels = [];

function percent(value) {
  if (value === null || value === undefined) return "—";
  return `${Math.round(value * 100)}%`;
}

function variantName(value) {
  const names = {
    camera: "Camera optimized",
    camera_fast: "Camera optimized",
    camera_robust: "Camera optimized",
    fast: "Fast analysis",
    full: "Detailed analysis",
    regular: "Detailed analysis",
  };
  return names[String(value || "").toLowerCase()] || String(value || "Standard analysis");
}

function splitDiagnosis(label) {
  const [plant, ...conditionParts] = String(label || "Unknown condition").split(" - ");
  return {
    plant: plant || "Plant",
    condition: conditionParts.join(" - ") || plant || "Unknown condition",
  };
}

function isHealthyLabel(label) {
  return String(label || "").toLowerCase().includes("healthy");
}

function showEmptyResult(statusText = "Awaiting image") {
  emptyResult.hidden = false;
  resultContent.hidden = true;
  resultStatus.textContent = statusText;
  resultStatus.className = "result-status is-waiting";
  confidenceBar.style.width = "0%";
  confidenceTrack.setAttribute("aria-valuenow", "0");
  topPredictions.replaceChildren();
}

function showError(message) {
  emptyResult.hidden = false;
  resultContent.hidden = true;
  emptyResult.querySelector("h3").textContent = "We could not analyze this image";
  emptyResult.querySelector("p").textContent = message;
  resultStatus.textContent = "Needs attention";
  resultStatus.className = "result-status is-error";
}

function restoreEmptyCopy() {
  emptyResult.querySelector("h3").textContent = "Ready when you are";
  emptyResult.querySelector("p").textContent =
    "Add a leaf photo to see the predicted condition, confidence score, and closest matches.";
}

function setLoading(isLoading) {
  analysisOverlay.hidden = !isLoading;
  dropZone.setAttribute("aria-busy", String(isLoading));
  modelSelect.disabled = isLoading;
  startBtn.disabled = isLoading;
  clearBtn.disabled = isLoading;

  if (isLoading) {
    emptyResult.hidden = false;
    resultContent.hidden = true;
    resultStatus.textContent = "Analyzing";
    resultStatus.className = "result-status is-loading";
    emptyResult.querySelector("h3").textContent = "Analyzing your leaf";
    emptyResult.querySelector("p").textContent =
      "The model is comparing color, texture, and spot patterns.";
  }
}

function updateModelDetails() {
  const selected = availableModels.find((model) => model.file === modelSelect.value);
  if (!selected) return;
  modelName.textContent = selected.name;
  const score = selected.metrics?.f1_weighted;
  const metric = typeof score === "number" ? ` · validation F1 ${percent(score)}` : "";
  modelVariant.textContent = `${variantName(selected.variant)} · ${selected.feature_count} features${metric}`;
}

async function loadModels() {
  const response = await fetch("/api/models");
  if (!response.ok) throw new Error("The analysis models could not be loaded.");

  const payload = await response.json();
  availableModels = payload.models || [];
  modelSelect.replaceChildren();

  if (!availableModels.length) {
    const option = document.createElement("option");
    option.textContent = "No model available";
    option.value = "";
    modelSelect.appendChild(option);
    modelSelect.disabled = true;
    modelName.textContent = "No analysis model found";
    modelVariant.textContent = "Add a model bundle to the models folder";
    showError("No trained model is available for analysis.");
    return;
  }

  availableModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.file;
    option.textContent = `${model.name} · ${variantName(model.variant)}`;
    modelSelect.appendChild(option);
  });

  modelSelect.value = payload.default_model || availableModels[0].file;
  updateModelDetails();
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  cameraStage.hidden = true;
  dropZone.hidden = false;
  captureBtn.hidden = true;
  captureBtn.disabled = true;
  startBtn.textContent = "Use camera";
}

async function startCamera() {
  if (stream) {
    stopCamera();
    return;
  }

  dropZone.hidden = true;
  cameraStage.hidden = false;
  captureBtn.hidden = false;
  emptyState.hidden = false;
  startBtn.textContent = "Close camera";

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    emptyState.hidden = true;
    captureBtn.disabled = false;
  } catch (error) {
    stopCamera();
    showError("Camera access was unavailable. Check your browser permission or upload a photo instead.");
  }
}

function reticleCropFromCanvas(sourceCanvas) {
  const insetX = sourceCanvas.width * 0.09;
  const insetY = sourceCanvas.height * 0.09;
  const cropWidth = sourceCanvas.width - insetX * 2;
  const cropHeight = sourceCanvas.height - insetY * 2;

  canvas.width = Math.round(cropWidth);
  canvas.height = Math.round(cropHeight);
  const outputContext = canvas.getContext("2d");
  outputContext.drawImage(
    sourceCanvas,
    Math.round(insetX),
    Math.round(insetY),
    canvas.width,
    canvas.height,
    0,
    0,
    canvas.width,
    canvas.height
  );

  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
}

function showPreview(blob) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(blob);
  imagePreview.src = previewUrl;
  imagePreview.hidden = false;
  uploadPrompt.hidden = true;
  dropZone.hidden = false;
  dropZone.classList.add("has-image");
  clearBtn.hidden = false;
  uploadButtonText.textContent = "Choose another photo";
}

function imageFileFromItems(items) {
  return Array.from(items || [])
    .find((item) => item.kind === "file" && item.type.startsWith("image/"))
    ?.getAsFile();
}

function drawVideoAsDisplayed() {
  const displayWidth = Math.round(video.clientWidth || video.videoWidth);
  const displayHeight = Math.round(video.clientHeight || video.videoHeight);
  const displayCanvas = document.createElement("canvas");
  displayCanvas.width = displayWidth;
  displayCanvas.height = displayHeight;
  const context = displayCanvas.getContext("2d");

  const sourceRatio = video.videoWidth / video.videoHeight;
  const displayRatio = displayWidth / displayHeight;
  let sourceX = 0;
  let sourceY = 0;
  let sourceWidth = video.videoWidth;
  let sourceHeight = video.videoHeight;

  if (sourceRatio > displayRatio) {
    sourceWidth = video.videoHeight * displayRatio;
    sourceX = (video.videoWidth - sourceWidth) / 2;
  } else {
    sourceHeight = video.videoWidth / displayRatio;
    sourceY = (video.videoHeight - sourceHeight) / 2;
  }

  context.drawImage(video, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, displayWidth, displayHeight);
  return displayCanvas;
}

function fileToCroppedBlob(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const imageUrl = URL.createObjectURL(file);

    image.onload = async () => {
      const sourceCanvas = document.createElement("canvas");
      sourceCanvas.width = image.naturalWidth;
      sourceCanvas.height = image.naturalHeight;
      sourceCanvas.getContext("2d").drawImage(image, 0, 0);
      URL.revokeObjectURL(imageUrl);
      resolve(await reticleCropFromCanvas(sourceCanvas));
    };
    image.onerror = () => {
      URL.revokeObjectURL(imageUrl);
      reject(new Error("The image could not be read."));
    };
    image.src = imageUrl;
  });
}

function renderResult(payload) {
  const diagnosis = splitDiagnosis(payload.label);
  const healthy = isHealthyLabel(payload.label);

  emptyResult.hidden = true;
  resultContent.hidden = false;
  resultPlant.textContent = diagnosis.plant;
  resultLabel.textContent = diagnosis.condition;
  resultSummary.textContent = healthy
    ? "The visible patterns most closely match a healthy leaf in the trained dataset."
    : `The visible patterns most closely match ${diagnosis.condition.toLowerCase()} in the trained dataset.`;

  resultStatus.textContent = healthy ? "Healthy pattern" : "Disease pattern";
  resultStatus.className = `result-status ${healthy ? "is-healthy" : "is-disease"}`;

  const confidenceValue = Math.round((payload.confidence || 0) * 100);
  confidenceText.textContent = percent(payload.confidence);
  confidenceBar.style.width = `${confidenceValue}%`;
  confidenceTrack.setAttribute("aria-valuenow", String(confidenceValue));

  topPredictions.replaceChildren();
  (payload.top_predictions || []).forEach((item) => {
    const row = document.createElement("div");
    const label = document.createElement("strong");
    const score = document.createElement("span");
    row.className = "prediction-row";
    label.textContent = item.label;
    score.textContent = percent(item.confidence);
    row.append(label, score);
    topPredictions.appendChild(row);
  });

  recommendationTitle.textContent = healthy ? "Keep monitoring" : "Inspect and isolate";
  recommendationText.textContent = healthy
    ? "Continue routine care and screen again if new spots, discoloration, or wilting appear."
    : "Separate affected foliage where practical and confirm the result with a local plant-care expert before treatment.";

  modelName.textContent = payload.model;
  modelVariant.textContent = variantName(payload.variant);
}

async function predictBlob(blob) {
  if (!blob || !modelSelect.value) return;
  currentBlob = blob;
  showPreview(blob);
  setLoading(true);

  const form = new FormData();
  form.append("model", modelSelect.value);
  form.append("image", blob, "leaf-capture.jpg");

  try {
    const response = await fetch("/api/predict", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The prediction could not be completed.");
    renderResult(payload);
  } catch (error) {
    showError(error.message || "The prediction could not be completed. Please try another image.");
  } finally {
    setLoading(false);
  }
}

async function handleImageFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Please choose an image file in JPG, PNG, or WEBP format.");
    return;
  }

  try {
    stopCamera();
    await predictBlob(await fileToCroppedBlob(file));
  } catch (error) {
    showError(error.message || "The image could not be prepared for analysis.");
  }
}

function clearImage() {
  stopCamera();
  currentBlob = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  imagePreview.removeAttribute("src");
  imagePreview.hidden = true;
  uploadPrompt.hidden = false;
  dropZone.classList.remove("has-image");
  clearBtn.hidden = true;
  uploadButtonText.textContent = "Choose photo";
  restoreEmptyCopy();
  showEmptyResult();
  updateModelDetails();
}

startBtn.addEventListener("click", startCamera);

captureBtn.addEventListener("click", async () => {
  if (!stream) return;
  const blob = await reticleCropFromCanvas(drawVideoAsDisplayed());
  stopCamera();
  await predictBlob(blob);
});

clearBtn.addEventListener("click", clearImage);

modelSelect.addEventListener("change", async () => {
  updateModelDetails();
  if (currentBlob) await predictBlob(currentBlob);
});

uploadInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (file) await handleImageFile(file);
  uploadInput.value = "";
});

dropZone.addEventListener("click", () => {
  if (!analysisOverlay.hidden) return;
  uploadInput.click();
});

dropZone.addEventListener("keydown", (event) => {
  if ((event.key === "Enter" || event.key === " ") && analysisOverlay.hidden) {
    event.preventDefault();
    uploadInput.click();
  }
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("is-dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("is-dragging");
});

dropZone.addEventListener("drop", async (event) => {
  event.preventDefault();
  dropZone.classList.remove("is-dragging");
  const file = imageFileFromItems(event.dataTransfer.items) || event.dataTransfer.files[0];
  await handleImageFile(file);
});

document.addEventListener("paste", async (event) => {
  const file = imageFileFromItems(event.clipboardData.items);
  if (file) await handleImageFile(file);
});

window.addEventListener("beforeunload", () => {
  stopCamera();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});

loadModels().catch((error) => showError(error.message || "The analysis models could not be loaded."));
