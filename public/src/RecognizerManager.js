// ═══════════════════════════════════════════════════════════════════════════
// RECOGNIZER MANAGER - Fixed to show words instead of numbers
// Uses MLP model from sign_language_web_ready for proper word mapping
// ═══════════════════════════════════════════════════════════════════════════

const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const loadingOverlay = document.getElementById('loading');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// UI Elements
const wordEnEl = document.getElementById('word-en');
const wordArEl = document.getElementById('word-ar');
const confFillEl = document.getElementById('conf-fill');
const confTextEl = document.getElementById('conf-text');
const sentenceBox = document.getElementById('sentence-box');

// State
let sentence = [];
let lastWord = "";
let modelData = null;
let tfModel = null;
let wordMapping = {}; // Index to Word
let predictionBuffer = [];
const BUFFER_SIZE = 5;
let cooldown = 0;

// Word Database with Arabic translations (Fixed - not dependent on dictionary.json)
const WORD_DB = {
    'ANCLE': { arabic: 'عم' },
    'BROTHER': { arabic: 'الأخ' },
    'EASY': { arabic: 'سهل' },
    'ENGAGEMENT': { arabic: 'الخطوبة' },
    'FAMILY': { arabic: 'العائلة' },
    'FATHER': { arabic: 'الأب' },
    'HIM': { arabic: 'هو' },
    'HOUR': { arabic: 'ساعه' },
    'HOW': { arabic: 'كيف' },
    'MINE': { arabic: 'حقي' },
    'MOTHER': { arabic: 'الأم' },
    'MOUNTH': { arabic: 'شهر' },
    'NAME': { arabic: 'الاسم' },
    'NO': { arabic: 'لا' },
    'PERCENTAGE': { arabic: 'مئه بالمئه' },
    'RAEDY': { arabic: 'جاهز' },
    'WHAT': { arabic: 'ماذا' },
    'WHEN': { arabic: 'متى' },
    'WHERE': { arabic: 'أين' },
    'YES': { arabic: 'نعم' },
    'cancer': { arabic: 'سرطان' },
    'cold': { arabic: 'برد' },
    'eat': { arabic: 'أكل' },
    'face': { arabic: 'وجه' },
    'fever': { arabic: 'حمى' },
    'loss of hair': { arabic: 'تساقط الشعر' },
    'medicine': { arabic: 'دواء' },
    'muscle': { arabic: 'عضلة' }
};

// Load Model - Now using MLP model with proper word_map
async function loadModel() {
    statusText.innerText = "Loading AI Model...";
    try {
        // Load MLP model from sign_language_web_ready (has proper word_map!)
        const response = await fetch('sign_language_web_ready/js/mlp_tfjs.json');
        modelData = await response.json();

        // Reconstruct Model in TF.js
        reconstructModel(modelData);

        // Setup Word Mapping from model's word_map
        wordMapping = modelData.word_map;

        console.log("✅ Model Loaded & Reconstructed");
        console.log("   Features:", modelData.input_len);
        console.log("   Classes:", Object.keys(wordMapping).length);
        console.log("   Words:", Object.values(wordMapping).slice(0, 5).join(', ') + '...');

        statusDot.classList.add('connected');
        statusText.innerText = "Ready (Offline AI)";
        loadingOverlay.style.display = 'none';

    } catch (e) {
        console.error("Error loading model:", e);
        statusText.innerText = "Error Loading Model";
    }
}

function reconstructModel(data) {
    tfModel = tf.sequential();

    // MLP model structure: 4 layers (256 -> 128 -> 64 -> output)
    tfModel.add(tf.layers.dense({
        units: 256,
        inputShape: [data.input_len],
        activation: 'relu',
        weights: [tf.tensor(data.weights[0]), tf.tensor(data.biases[0])],
        trainable: false
    }));

    tfModel.add(tf.layers.dense({
        units: 128,
        activation: 'relu',
        weights: [tf.tensor(data.weights[1]), tf.tensor(data.biases[1])],
        trainable: false
    }));

    tfModel.add(tf.layers.dense({
        units: 64,
        activation: 'relu',
        weights: [tf.tensor(data.weights[2]), tf.tensor(data.biases[2])],
        trainable: false
    }));

    tfModel.add(tf.layers.dense({
        units: data.biases[3].length,
        activation: 'softmax',
        weights: [tf.tensor(data.weights[3]), tf.tensor(data.biases[3])],
        trainable: false
    }));
}

// ---------------- FEATURE EXTRACTION (EXACT MATCH WITH PYTHON) ----------------

function normalizeKeypoints(kpts) {
    if (!kpts || kpts.length === 0) return Array(0).fill(0);

    let cx = 0, cy = 0, cz = 0;
    kpts.forEach(p => { cx += p.x; cy += p.y; cz += p.z; });
    cx /= kpts.length; cy /= kpts.length; cz /= kpts.length;

    const centered = kpts.map(p => ({ x: p.x - cx, y: p.y - cy, z: p.z - cz }));

    let maxDist = 0;
    centered.forEach(p => {
        const d = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (d > maxDist) maxDist = d;
    });
    if (maxDist === 0) maxDist = 1;

    return centered.flatMap(p => [p.x / maxDist, p.y / maxDist, p.z / maxDist]);
}

function computeFingerAngles(pts) {
    const tips = [4, 8, 12, 16, 20];
    const dips = [3, 7, 11, 15, 19];
    const mcps = [2, 5, 9, 13, 17];
    const angles = [];

    for (let i = 1; i < 5; i++) {
        const a = pts[mcps[i]];
        const b = pts[dips[i]];
        const c = pts[tips[i]];

        const v1 = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
        const v2 = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };

        const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
        const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);

        let cosine = dot / ((mag1 * mag2) + 1e-6);
        cosine = Math.max(-1, Math.min(1, cosine));
        angles.push(Math.acos(cosine) * (180 / Math.PI));
    }
    return angles;
}

function buildFeatures(results) {
    let handLms = null;
    let label = '';

    // MediaPipe swaps hands in mirror mode
    if (results.rightHandLandmarks) {
        handLms = results.rightHandLandmarks;
        label = 'left';
    } else if (results.leftHandLandmarks) {
        handLms = results.leftHandLandmarks;
        label = 'right';
    }

    if (!handLms) return null;

    const pts = handLms.map(l => ({ x: l.x, y: l.y, z: l.z }));

    // Static Features (82 total)
    const handVec = normalizeKeypoints(pts); // 63
    const angs = computeFingerAngles(pts);   // 4

    let staticFeats = [...handVec, ...angs]; // 67

    // Arm features from pose if available
    let armFeats = Array(9).fill(0);
    let distFeats = Array(3).fill(0);

    if (results.poseLandmarks) {
        const pose = results.poseLandmarks;
        const wrist = pts[0];

        // Get arm points
        const armKeys = label === 'left'
            ? [11, 13, 15] // LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
            : [12, 14, 16]; // RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST

        const armPts = armKeys.map(idx => pose[idx] ? { x: pose[idx].x, y: pose[idx].y, z: pose[idx].z } : { x: 0, y: 0, z: 0 });
        armFeats = normalizeKeypoints(armPts);

        // Nose to wrist distance
        const nose = pose[0] || { x: 0, y: 0, z: 0 };
        distFeats = [wrist.x - nose.x, wrist.y - nose.y, wrist.z - nose.z];
    }

    staticFeats.push(...armFeats, ...distFeats); // 67 + 9 + 3 = 79

    // Pad to 82
    while (staticFeats.length < 82) staticFeats.push(0);
    if (staticFeats.length > 82) staticFeats = staticFeats.slice(0, 82);

    // Dynamic Features (21 total) - zeros for now
    const dynamicFeats = Array(21).fill(0);

    return [...staticFeats, ...dynamicFeats]; // 103 total
}

// ---------------- INFERENCE LOOP ----------------

async function predict(features) {
    if (!tfModel || !modelData) return;

    tf.tidy(() => {
        let input = tf.tensor2d([features]);

        // Scale using MLP scaler (mean/std)
        const mean = tf.tensor1d(modelData.scaler.mean);
        const std = tf.tensor1d(modelData.scaler.std);
        input = input.sub(mean).div(std.add(1e-6));

        // Predict
        const output = tfModel.predict(input);
        const probs = output.dataSync();

        // Argmax
        let maxIdx = 0;
        let maxProb = -1;
        for (let i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIdx = i;
            }
        }

        const word = wordMapping[maxIdx]; // e.g. "FAMILY"
        handlePrediction(word, maxProb);
    });
}

function handlePrediction(word, confidence) {
    // Get Arabic translation
    const arabic = WORD_DB[word] ? WORD_DB[word].arabic : word;

    wordEnEl.innerText = word;
    wordArEl.innerText = arabic;

    const confPercent = Math.round(confidence * 100);
    confFillEl.style.width = `${confPercent}%`;
    confTextEl.innerText = `${confPercent}% Confidence`;

    // Determine Language from Global UI
    const currentLang = window.globalUI ? window.globalUI.currentLang : 'ar';
    const textToUse = currentLang === 'en' ? word : arabic;

    // Smoothing with better threshold
    if (confidence > 0.5) {
        if (cooldown > 0) {
            cooldown--;
            return;
        }

        predictionBuffer.push(word);
        if (predictionBuffer.length > BUFFER_SIZE) predictionBuffer.shift();

        const count = predictionBuffer.filter(w => w === word).length;
        if (count >= 3 && word !== lastWord) {
            lastWord = word;
            sentence.push(textToUse);
            updateSentenceBox();
            speak(textToUse);
            cooldown = 10;
            predictionBuffer = [];
        }
    }
}

// ---------------- SENTENCE MANAGEMENT ----------------

function updateSentenceBox() {
    sentenceBox.innerText = sentence.join(' ');
}

function clearSentence() {
    sentence = [];
    lastWord = "";
    // Update placeholder based on language if possible, but static text for now
    const currentLang = window.globalUI ? window.globalUI.currentLang : 'ar';
    sentenceBox.innerText = currentLang === 'en' ? "Waiting for signs..." : "في انتظار الإشارات...";
}

function speakSentence() {
    const text = sentence.join(' ');
    if (text) speak(text);
}

function formSentenceAI() {
    // AI sentence formation - could use an API
    const text = sentence.join(' ');
    if (text) {
        sentenceBox.innerText = `📝 ${text}`;
    }
}

function speak(text) {
    if ('speechSynthesis' in window) {
        const currentLang = window.globalUI ? window.globalUI.currentLang : 'ar';
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = currentLang === 'ar' ? 'ar-SA' : 'en-US';
        speechSynthesis.speak(utterance);
    }
}

// ---------------- MEDIAPIPE HOLISTIC ----------------

const holistic = new Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});

holistic.setOptions({
    modelComplexity: 0, // Lite for speed
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    refineFaceLandmarks: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

holistic.onResults(onResults);

function onResults(results) {
    // Draw
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    // Draw pose (body skeleton)
    if (results.poseLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00CEFF', lineWidth: 4 });
        drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#FF0099', lineWidth: 2, radius: 5 });
    }

    // Draw hands
    if (results.rightHandLandmarks) {
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 3 });
        drawLandmarks(canvasCtx, results.rightHandLandmarks, { color: '#00FF00', lineWidth: 1, radius: 5 });
    }
    if (results.leftHandLandmarks) {
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#FF0000', lineWidth: 3 });
        drawLandmarks(canvasCtx, results.leftHandLandmarks, { color: '#FF0000', lineWidth: 1, radius: 5 });
    }

    canvasCtx.restore();

    // Extract & Predict
    const features = buildFeatures(results);
    if (features) {
        predict(features);
    }
}

// ---------------- CAMERA ----------------

// ---------------- CAMERA ----------------
let camera;
let currentFacingMode = 'user';

function isMobile() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

// Show button on mobile
if (isMobile()) {
    const switchBtn = document.getElementById('camera-switch-btn');
    if (switchBtn) switchBtn.style.display = 'flex';
}

window.switchCamera = async function () {
    currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    if (camera) {
        await camera.stop();
        camera = null;
    }
    startCamera();
};

function startCamera() {
    camera = new Camera(videoElement, {
        onFrame: async () => {
            await holistic.send({ image: videoElement });
        },
        width: 640,
        height: 480,
        facingMode: currentFacingMode
    });
    camera.start();
}

// Initialize
async function init() {
    await loadModel();
    startCamera();
}

init();
