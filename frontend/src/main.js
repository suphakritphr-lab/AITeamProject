import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "@mediapipe/tasks-vision";

import "./style.css";

const API_URL = `${window.location.protocol}//${window.location.hostname}:8000`;

let poseLandmarker = null;
let cameraStream = null;
let cameraRunning = false;
let lastVideoTime = -1;
let lastPredictionTime = 0;
let currentFeatures = null;

// Elements
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const imageInput = document.getElementById("imageInput");
const uploadLabel = document.getElementById("uploadLabel");

const cameraTab = document.getElementById("cameraTab");
const uploadTab = document.getElementById("uploadTab");

const startCameraButton = document.getElementById("startCameraButton");
const stopCameraButton = document.getElementById("stopCameraButton");

const message = document.getElementById("message");
const detectionBadge = document.getElementById("detectionBadge");
const emptyState = document.getElementById("emptyState");

const postureResult = document.getElementById("postureResult");
const postureLabel = document.getElementById("postureLabel");
const confidenceText = document.getElementById("confidenceText");
const confidencePercent = document.getElementById("confidencePercent");
const confidenceBar = document.getElementById("confidenceBar");
const probabilityList = document.getElementById("probabilityList");
const featureGrid = document.getElementById("featureGrid");
const apiStatus = document.getElementById("apiStatus");

// --------------------------------------------------
// MediaPipe initialization
// --------------------------------------------------

async function initializePose() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        poseLandmarker = await PoseLandmarker.createFromOptions(
            vision,
            {
                baseOptions: {
                    modelAssetPath:
                        "https://storage.googleapis.com/mediapipe-models/" +
                        "pose_landmarker/pose_landmarker_lite/" +
                        "float16/1/pose_landmarker_lite.task"
                },
                runningMode: "VIDEO",
                numPoses: 1,
                minPoseDetectionConfidence: 0.5,
                minPosePresenceConfidence: 0.5,
                minTrackingConfidence: 0.5
            }
        );

        message.textContent = "Ready. Start the camera or upload an image.";
        detectionBadge.textContent = "Ready";
        detectionBadge.className = "detection-badge ready";

        await checkApi();

    } catch (error) {
        console.error(error);
        message.textContent = "Could not initialize MediaPipe.";
        detectionBadge.textContent = "Error";
        detectionBadge.className = "detection-badge error";
    }
}

initializePose();

// --------------------------------------------------
// API status
// --------------------------------------------------

async function checkApi() {
    try {
        const response = await fetch(`${API_URL}/`, { cache: "no-store" });

        if (!response.ok) throw new Error("API error");

        apiStatus.innerHTML = '<span class="status-dot"></span> AI server online';
        apiStatus.className = "status-pill status-online";
    } catch (error) {
        apiStatus.innerHTML = '<span class="status-dot"></span> AI server offline';
        apiStatus.className = "status-pill status-offline";
    }
}

// --------------------------------------------------
// Camera mode
// --------------------------------------------------

cameraTab.addEventListener("click", () => {
    setMode("camera");
});

uploadTab.addEventListener("click", () => {
    setMode("upload");
});

async function setPoseMode(mode) {
    if (!poseLandmarker) return;

    await poseLandmarker.setOptions({
        runningMode: mode
    });
}

function setMode(mode) {
    if (mode === "camera") {
        cameraTab.classList.add("active");
        uploadTab.classList.remove("active");
        uploadLabel.classList.add("hidden");

        emptyState.classList.remove("hidden");
        message.textContent = "Start the camera to analyze your posture.";
    } else {
        uploadTab.classList.add("active");
        cameraTab.classList.remove("active");
        uploadLabel.classList.remove("hidden");

        stopCamera();
        emptyState.classList.remove("hidden");
        message.textContent = "Choose an image containing one sitting person.";
    }
}

startCameraButton.addEventListener("click", startCamera);
stopCameraButton.addEventListener("click", stopCamera);

async function startCamera() {
    if (!poseLandmarker) {
        message.textContent = "MediaPipe is still loading...";
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();

        cameraRunning = true;
        lastVideoTime = -1;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        emptyState.classList.add("hidden");

        startCameraButton.disabled = true;
        stopCameraButton.disabled = false;

        detectionBadge.textContent = "Scanning";
        detectionBadge.className = "detection-badge scanning";
        message.textContent = "Looking for a person...";

        requestAnimationFrame(processCameraFrame);

    } catch (error) {
        console.error(error);
        message.textContent =
            "Camera access failed. Check browser permissions and make sure the page is served over HTTP/HTTPS.";
    }
}

function stopCamera() {
    cameraRunning = false;

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    video.srcObject = null;

    startCameraButton.disabled = false;
    stopCameraButton.disabled = true;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detectionBadge.textContent = "Stopped";
    detectionBadge.className = "detection-badge";
    message.textContent = "Camera stopped.";
}

async function processCameraFrame(timestamp) {
    if (!cameraRunning) return;

    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;

        try {
            await setPoseMode("VIDEO");
            const detection = poseLandmarker.detectForVideo(video, timestamp);
            await handleDetection(detection, true);
        } catch (error) {
            console.error(error);
        }
    }

    requestAnimationFrame(processCameraFrame);
}

// --------------------------------------------------
// Image upload
// --------------------------------------------------

imageInput.addEventListener("change", async event => {
    const file = event.target.files[0];

    if (!file) return;

    stopCamera();

    const image = new Image();

    image.onload = async () => {
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0);

        emptyState.classList.add("hidden");

        detectionBadge.textContent = "Scanning";
        detectionBadge.className = "detection-badge scanning";
        message.textContent = "Detecting pose...";

        try {
            // IMAGE mode is not active after camera mode, so use a fresh
            // IMAGE-mode detector for uploaded images.
            await setPoseMode("IMAGE");
            const detection = poseLandmarker.detect(image);
            await handleDetection(detection, false);
        } catch (error) {
            console.error(error);
            message.textContent = "Could not analyze this image.";
        }

        URL.revokeObjectURL(image.src);
    };

    image.src = URL.createObjectURL(file);
});

// --------------------------------------------------
// Detection handling
// --------------------------------------------------

async function handleDetection(detection, allowPrediction) {
    if (!detection.landmarks || detection.landmarks.length === 0) {
        currentFeatures = null;

        detectionBadge.textContent = "No person";
        detectionBadge.className = "detection-badge error";
        message.textContent = "No person detected. Adjust the camera/image.";

        clearPrediction();
        return;
    }

    const landmarks = detection.landmarks[0];

    // Draw image/camera frame first, then skeleton.
    if (allowPrediction) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    }

    const drawingUtils = new DrawingUtils(ctx);

    drawingUtils.drawConnectors(
        landmarks,
        PoseLandmarker.POSE_CONNECTIONS,
        { lineWidth: 2 }
    );

    drawingUtils.drawLandmarks(
        landmarks,
        { radius: 4 }
    );

    try {
        currentFeatures = calculateFeatures(landmarks);
    } catch (error) {
        console.error(error);
        detectionBadge.textContent = "Incomplete pose";
        detectionBadge.className = "detection-badge error";
        message.textContent =
            "Some body landmarks are missing. Please move so your body is visible.";
        return;
    }

    updateFeatureDisplay(currentFeatures);

    detectionBadge.textContent = "Person detected";
    detectionBadge.className = "detection-badge ready";
    message.textContent = "Pose detected. Classifying...";

    // Avoid flooding FastAPI from the camera.
    const now = performance.now();

    if (allowPrediction && now - lastPredictionTime < 250) {
        return;
    }

    lastPredictionTime = now;

    await predictPosture(currentFeatures);
}

// --------------------------------------------------
// Vector math
// Same feature definitions as the uploaded training code.
// --------------------------------------------------

function angleBetween(a, b, c) {
    const ba = {
        x: a.x - b.x,
        y: a.y - b.y
    };

    const bc = {
        x: c.x - b.x,
        y: c.y - b.y
    };

    const dot =
        ba.x * bc.x +
        ba.y * bc.y;

    const magnitudeBA =
        Math.sqrt(
            ba.x * ba.x +
            ba.y * ba.y
        );

    const magnitudeBC =
        Math.sqrt(
            bc.x * bc.x +
            bc.y * bc.y
        );

    if (magnitudeBA === 0 || magnitudeBC === 0) {
        throw new Error("Invalid landmark geometry");
    }

    const cosine =
        dot /
        (magnitudeBA * magnitudeBC);

    const clamped =
        Math.max(-1, Math.min(1, cosine));

    return (
        Math.acos(clamped)
        * 180
        / Math.PI
    );
}

function distance(a, b) {
    return Math.sqrt(
        Math.pow(a.x - b.x, 2) +
        Math.pow(a.y - b.y, 2)
    );
}

// --------------------------------------------------
// Feature extraction
// EXACTLY matches the uploaded main.js.
// --------------------------------------------------

function calculateFeatures(landmarks) {
    const nose = landmarks[0];

    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];

    const leftElbow = landmarks[13];
    const rightElbow = landmarks[14];

    const leftWrist = landmarks[15];
    const rightWrist = landmarks[16];

    const leftHip = landmarks[23];
    const rightHip = landmarks[24];

    const leftKnee = landmarks[25];
    const rightKnee = landmarks[26];

    const required = [
        nose,
        leftShoulder,
        rightShoulder,
        leftElbow,
        rightElbow,
        leftWrist,
        rightWrist,
        leftHip,
        rightHip,
        leftKnee,
        rightKnee,
        landmarks[27],
        landmarks[28]
    ];

    if (required.some(point => !point)) {
        throw new Error("Missing required landmarks");
    }

    const shoulderCenter = {
        x: (leftShoulder.x + rightShoulder.x) / 2,
        y: (leftShoulder.y + rightShoulder.y) / 2
    };

    const hipCenter = {
        x: (leftHip.x + rightHip.x) / 2,
        y: (leftHip.y + rightHip.y) / 2
    };

    const leftElbowAngle =
        angleBetween(
            leftShoulder,
            leftElbow,
            leftWrist
        );

    const rightElbowAngle =
        angleBetween(
            rightShoulder,
            rightElbow,
            rightWrist
        );

    const leftKneeAngle =
        angleBetween(
            leftHip,
            leftKnee,
            landmarks[27]
        );

    const rightKneeAngle =
        angleBetween(
            rightHip,
            rightKnee,
            landmarks[28]
        );

    const shoulderAngle =
        Math.atan2(
            rightShoulder.y - leftShoulder.y,
            rightShoulder.x - leftShoulder.x
        ) * 180 / Math.PI;

    const torsoAngle =
        Math.atan2(
            hipCenter.x - shoulderCenter.x,
            hipCenter.y - shoulderCenter.y
        ) * 180 / Math.PI;

    const headShoulderDistance =
        distance(nose, shoulderCenter);

    const shoulderHipDistance =
        distance(shoulderCenter, hipCenter);

    if (shoulderHipDistance === 0) {
        throw new Error("Invalid shoulder/hip distance");
    }

    const normalizedHeadDistance =
        headShoulderDistance /
        shoulderHipDistance;

    return {
        nose_x: nose.x,
        nose_y: nose.y,

        left_shoulder_x: leftShoulder.x,
        left_shoulder_y: leftShoulder.y,

        right_shoulder_x: rightShoulder.x,
        right_shoulder_y: rightShoulder.y,

        left_elbow_angle: leftElbowAngle,
        right_elbow_angle: rightElbowAngle,

        left_knee_angle: leftKneeAngle,
        right_knee_angle: rightKneeAngle,

        shoulder_angle: shoulderAngle,
        torso_angle: torsoAngle,

        normalized_head_distance: normalizedHeadDistance
    };
}

// --------------------------------------------------
// Prediction
// --------------------------------------------------

async function predictPosture(features) {
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(features)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText);
        }

        const data = await response.json();

        displayPrediction(data);

        apiStatus.innerHTML = '<span class="status-dot"></span> AI server online';
        apiStatus.className = "status-pill status-online";

    } catch (error) {
        console.error(error);

        apiStatus.innerHTML = '<span class="status-dot"></span> AI server error';
        apiStatus.className = "status-pill status-offline";

        message.textContent =
            "Pose detected, but the Python AI server could not be reached.";
    }
}

function displayPrediction(data) {
    const posture = String(data.posture || "unknown");
    const confidence = Number(data.confidence || 0);

    postureLabel.textContent = formatPosture(posture);
    confidenceText.textContent =
        `${Math.round(confidence * 100)}% model confidence`;

    confidencePercent.textContent =
        `${Math.round(confidence * 100)}%`;

    confidenceBar.style.width =
        `${Math.max(0, Math.min(100, confidence * 100))}%`;

    postureResult.className =
        `posture-result ${getPostureClass(posture)}`;

    message.textContent =
        `Prediction: ${formatPosture(posture)}`;

    const probabilities = data.probabilities || {};

    const entries = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1]);

    if (entries.length === 0) {
        probabilityList.innerHTML =
            '<div class="prob-empty">Probability data unavailable</div>';
        return;
    }

    probabilityList.innerHTML = entries.map(([label, value]) => `
        <div class="probability-row">
            <div class="probability-name">
                <span>${escapeHtml(formatPosture(label))}</span>
                <span>${Math.round(value * 100)}%</span>
            </div>
            <div class="mini-track">
                <div class="mini-bar" style="width:${Math.round(value * 100)}%"></div>
            </div>
        </div>
    `).join("");
}

function clearPrediction() {
    postureLabel.textContent = "—";
    confidenceText.textContent = "Waiting for detection";
    confidencePercent.textContent = "0%";
    confidenceBar.style.width = "0%";
    postureResult.className = "posture-result neutral";
    probabilityList.innerHTML =
        '<div class="prob-empty">No prediction yet</div>';
    featureGrid.innerHTML =
        '<div class="feature-empty">No pose detected</div>';
}

function formatPosture(value) {
    return value
        .replaceAll("_", " ")
        .replace(/\b\w/g, c => c.toUpperCase());
}

function getPostureClass(value) {
    const v = value.toLowerCase();

    if (v === "good") return "good";
    if (v === "slouching") return "slouching";
    if (v === "forward_lean") return "forward-lean";

    return "unknown";
}

function updateFeatureDisplay(features) {
    const items = [
        ["Torso angle", features.torso_angle, "°"],
        ["Shoulder angle", features.shoulder_angle, "°"],
        ["Left elbow", features.left_elbow_angle, "°"],
        ["Right elbow", features.right_elbow_angle, "°"],
        ["Left knee", features.left_knee_angle, "°"],
        ["Right knee", features.right_knee_angle, "°"],
        ["Head distance", features.normalized_head_distance, ""]
    ];

    featureGrid.innerHTML = items.map(([name, value, unit]) => `
        <div class="feature-item">
            <span>${name}</span>
            <strong>${Number(value).toFixed(1)}${unit}</strong>
        </div>
    `).join("");
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
