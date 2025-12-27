/**
 * MLP Model Inference and Test Logic (Fixed: Holistic + UI Redesign)
 */

class MLPClassifier {
    constructor(modelData) {
        this.modelData = modelData;
        this.weights = modelData.layers.weights;
        this.biases = modelData.layers.biases;
        this.activations = modelData.layers.activations;
        this.mean = modelData.scaler.mean;
        this.scale = modelData.scaler.scale;
        this.classes = modelData.classes;
    }

    transform(features) {
        let feat = [...features];
        if (feat.length < this.mean.length) {
            feat = feat.concat(new Array(this.mean.length - feat.length).fill(0));
        } else if (feat.length > this.mean.length) {
            feat = feat.slice(0, this.mean.length);
        }
        return feat.map((val, i) => (val - this.mean[i]) / this.scale[i]);
    }

    predictProba(features) {
        let current = this.transform(features);
        for (let i = 0; i < this.weights.length; i++) {
            const W = this.weights[i];
            const b = this.biases[i];
            const activation = this.activations[i];

            const nextLayer = new Array(b.length).fill(0);
            for (let j = 0; j < b.length; j++) {
                let sum = 0;
                for (let k = 0; k < current.length; k++) {
                    sum += current[k] * W[k][j];
                }
                nextLayer[j] = sum + b[j];
            }

            if (activation === 'relu') {
                current = nextLayer.map(v => Math.max(0, v));
            } else if (activation === 'softmax') {
                const max = Math.max(...nextLayer);
                const exps = nextLayer.map(v => Math.exp(v - max));
                const sumExps = exps.reduce((a, b) => a + b, 0);
                current = exps.map(v => v / sumExps);
            } else {
                current = nextLayer;
            }
        }
        return current;
    }

    predict(features) {
        const probs = this.predictProba(features);
        let maxIdx = 0;
        let maxVal = -1;
        for (let i = 0; i < probs.length; i++) {
            if (probs[i] > maxVal) {
                maxVal = probs[i];
                maxIdx = i;
            }
        }
        return {
            label: this.classes[maxIdx],
            probability: maxVal,
            allProbs: probs
        };
    }
}

// Global state
let holistic, camera;
let mlp;
let prevWrist = { left: null, right: null };
let velBuffers = { left: [], right: [] };
const DYN_WINDOW = 8;
const SMOOTHING_FRAMES = 5;
const predictionBuffer = [];
let lastPredictionLabel = '';

// Arabic Mapping
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

// DOM Elements (Updated for new design)
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');

const wordEnEl = document.getElementById('word-en');
const wordArEl = document.getElementById('word-ar');
const confFillEl = document.getElementById('conf-fill');
const confTextEl = document.getElementById('conf-text');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');
const loadingEl = document.getElementById('loading');
const sentenceBox = document.getElementById('sentence-box');

async function loadModel() {
    loadingEl.style.display = 'flex';
    statusText.textContent = "Loading Model...";
    try {
        const response = await fetch('mlp_dynamic106_fixed/mlp_model.json');
        const modelData = await response.json();
        mlp = new MLPClassifier(modelData);
        console.log("Model Loaded:", modelData);
        loadingEl.style.display = 'none';
        statusText.textContent = "Model Ready | Waiting for Camera";
        statusDot.style.background = 'orange';
    } catch (e) {
        console.error(e);
        statusText.textContent = "Error Loading Model";
        statusDot.style.background = 'red';
    }
}

// Feature Extraction Utils (Same as before)
function normalizeKeypoints(kpts) {
    if (!kpts || kpts.length === 0) return [];
    let cx = 0, cy = 0, cz = 0;
    kpts.forEach(p => { cx += p.x; cy += p.y; cz += p.z; });
    cx /= kpts.length; cy /= kpts.length; cz /= kpts.length;

    const centered = kpts.map(p => ({ x: p.x - cx, y: p.y - cy, z: p.z - cz }));
    let maxDist = 0;
    centered.forEach(p => {
        const d = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (d > maxDist) maxDist = d;
    });
    maxDist += 1e-6;
    return centered.map(p => [p.x / maxDist, p.y / maxDist, p.z / maxDist]);
}

function computeFingerAngles(pts) {
    const tips = [4, 8, 12, 16, 20];
    const dips = [3, 7, 11, 15, 19];
    const mcps = [2, 5, 9, 13, 17];
    const angles = [];
    for (let i = 1; i < 5; i++) {
        const a = pts[mcps[i]]; const b = pts[dips[i]]; const c = pts[tips[i]];
        const ba = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
        const cb = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
        const dot = ba.x * cb.x + ba.y * cb.y + ba.z * cb.z;
        const normBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y + ba.z * ba.z);
        const normCB = Math.sqrt(cb.x * cb.x + cb.y * cb.y + cb.z * cb.z);
        let cosine = dot / (normBA * normCB + 1e-6);
        cosine = Math.max(-1, Math.min(1, cosine));
        angles.push(Math.acos(cosine) * 180 / Math.PI);
    }
    return angles;
}

function buildStaticFeatures(pts, arm, side) {
    const normalizedHand = normalizeKeypoints(pts).flat();
    const angs = computeFingerAngles(pts);
    const keys = side === 'left' ? [11, 13, 15] : [12, 14, 16];
    const armPts = keys.map(idx => {
        if (arm && arm[idx]) return { x: arm[idx].x, y: arm[idx].y, z: arm[idx].z };
        return { x: 0, y: 0, z: 0 };
    });
    const normalizedArm = normalizeKeypoints(armPts).flat();
    const noseIdx = 0;
    const nose = (arm && arm[noseIdx]) ? { x: arm[noseIdx].x, y: arm[noseIdx].y, z: arm[noseIdx].z } : { x: 0, y: 0, z: 0 };
    const wrist = pts[0];
    const distVec = [wrist.x - nose.x, wrist.y - nose.y, wrist.z - nose.z];

    let staticFeat = [...normalizedHand, ...angs, ...normalizedArm, ...distVec];
    if (staticFeat.length < 82) staticFeat = staticFeat.concat(new Array(82 - staticFeat.length).fill(0));
    return staticFeat;
}

function computeDynamicStats(buffer) {
    if (buffer.length === 0) return new Array(21).fill(0);
    const N = buffer.length;
    const getCol = (colIdx) => buffer.map(v => v[colIdx]);
    const calcStats = (arr) => {
        const n = arr.length;
        if (n === 0) return { mean: 0, std: 0, min: 0, max: 0, range: 0, median: 0, var: 0 };
        const mean = arr.reduce((a, b) => a + b, 0) / n;
        const variance = arr.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
        const std = Math.sqrt(variance);
        const min = Math.min(...arr);
        const max = Math.max(...arr);
        const range = max - min;
        const sorted = [...arr].sort((a, b) => a - b);
        const median = sorted[Math.floor(n / 2)];
        return { mean, std, min, max, range, median, var: variance };
    };
    const s0 = calcStats(getCol(0));
    const s1 = calcStats(getCol(1));
    const s2 = calcStats(getCol(2));

    // Flatten output
    const out = [];
    [s0, s1, s2].forEach(s => out.push(s.mean));
    [s0, s1, s2].forEach(s => out.push(s.std));
    [s0, s1, s2].forEach(s => out.push(s.min));
    [s0, s1, s2].forEach(s => out.push(s.max));
    [s0, s1, s2].forEach(s => out.push(s.range));
    [s0, s1, s2].forEach(s => out.push(s.median));
    [s0, s1, s2].forEach(s => out.push(s.var));
    return out;
}

// ----------------Detection Loop----------------
function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.poseLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: 'rgba(0,255,255,0.5)', lineWidth: 2 });
        drawLandmarks(canvasCtx, results.poseLandmarks, { color: 'rgba(255,0,0,0.5)', lineWidth: 1, radius: 2 });
    }
    if (results.leftHandLandmarks) {
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#CC0000', lineWidth: 2 });
        drawLandmarks(canvasCtx, results.leftHandLandmarks, { color: '#00FF00', lineWidth: 1, radius: 3 });
    }
    if (results.rightHandLandmarks) {
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#00CC00', lineWidth: 2 });
        drawLandmarks(canvasCtx, results.rightHandLandmarks, { color: '#FF0000', lineWidth: 1, radius: 3 });
    }

    if (!mlp) {
        canvasCtx.restore();
        return;
    }

    let poseDict = {};
    if (results.poseLandmarks) results.poseLandmarks.forEach((lm, i) => poseDict[i] = lm);

    // Logic: Left hand (visual) = User Right hand. MediaPipe Holistic "rightHandLandmarks" is actually user's LEFT hand in mirrored stream?
    // Let's stick to the previous swapping logic if it worked correctly.
    // Python script logic: "SWAP: MediaPipe's 'left' is actually user's right hand"
    // results.leftHandLandmarks -> Label 'right'
    // results.rightHandLandmarks -> Label 'left'

    let rightFeats = null;
    let leftFeats = null;

    const processHand = (landmarks, label) => {
        if (!landmarks) return null;
        const wrist = landmarks[0];
        const velVec = [0, 0, 0];
        const prev = prevWrist[label];
        if (prev) {
            velVec[0] = (wrist.x - prev.x) * 30;
            velVec[1] = (wrist.y - prev.y) * 30;
            velVec[2] = (wrist.z - prev.z) * 30;
        }
        prevWrist[label] = { ...wrist };
        if (!velBuffers[label]) velBuffers[label] = [];
        const buf = velBuffers[label];
        buf.push(velVec);
        if (buf.length > DYN_WINDOW) buf.shift();

        const staticF = buildStaticFeatures(landmarks, poseDict, label);
        const dynamicF = computeDynamicStats(buf);
        return [...staticF, ...dynamicF];
    };

    if (results.leftHandLandmarks) rightFeats = processHand(results.leftHandLandmarks, 'right');
    if (results.rightHandLandmarks) leftFeats = processHand(results.rightHandLandmarks, 'left');

    let candidates = [];
    const runPred = (feats, handLabel) => {
        if (feats) {
            const res = mlp.predict(feats);
            candidates.push({ ...res, handLabel });
        }
    };

    runPred(leftFeats, 'left');
    runPred(rightFeats, 'right');
    if (leftFeats && rightFeats) {
        const bothFeats = leftFeats.map((v, i) => (v + rightFeats[i]) / 2.0);
        runPred(bothFeats, 'both');
    }

    if (candidates.length > 0) {
        candidates.sort((a, b) => b.probability - a.probability);
        const best = candidates[0];

        predictionBuffer.push(best);
        if (predictionBuffer.length > SMOOTHING_FRAMES) predictionBuffer.shift();

        const correctCount = predictionBuffer.filter(p => p.label === best.label).length;

        // Show result
        if (best.probability > 0.5) { // Threshold
            wordEnEl.textContent = best.label;
            const arabic = WORD_DB[best.label] ? WORD_DB[best.label].arabic : best.label;
            wordArEl.textContent = arabic;

            const confVal = Math.round(best.probability * 100);
            confFillEl.style.width = `${confVal}%`;
            confTextEl.textContent = `${confVal}% Confidence`;

            statusText.textContent = `Active | Detected: ${best.label}`;

            // Append to history if stable change
            if (correctCount >= SMOOTHING_FRAMES * 0.8 && best.label !== lastPredictionLabel) {
                lastPredictionLabel = best.label;
                if (sentenceBox.textContent.includes('Waiting')) sentenceBox.textContent = "";
                sentenceBox.textContent += " " + arabic;

                // Speak the Arabic word
                speak(arabic, 'ar');
            }
        }
    } else {
        statusText.textContent = "Active | No Hand Detected";
        confFillEl.style.width = "0%";
    }

    canvasCtx.restore();
}

function speak(text, lang = 'ar') {
    if ('speechSynthesis' in window) {
        // Cancel previous speech to avoid lag
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = lang === 'ar' ? 'ar-SA' : 'en-US';
        utterance.rate = 0.9; // Slightly slower for clarity
        window.speechSynthesis.speak(utterance);
    }
}

// ----------------Initialization----------------

// ----------------Initialization----------------

holistic = new Holistic({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
    }
});

holistic.setOptions({
    modelComplexity: 0, // Lite mode for speed
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

holistic.onResults(onResults);

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
    await initCamera();
};

async function initCamera() {
    camera = new Camera(videoElement, {
        onFrame: async () => {
            await holistic.send({ image: videoElement });
        },
        width: 640,
        height: 480,
        facingMode: currentFacingMode
    });

    // If we are already running, start it immediately
    if (document.getElementById('btn-start').style.display === 'none') {
        await camera.start();
    }
}

// Start Button Logic
const startBtn = document.getElementById('btn-start');
startBtn.addEventListener('click', async () => {
    startBtn.disabled = true;
    startBtn.textContent = "Initializing AI Engine (This may take a moment)...";
    statusText.textContent = "Loading MediaPipe & Camera...";
    statusDot.style.background = 'yellow';

    // Tiny delay to let UI update before heavy blocking task
    await new Promise(r => setTimeout(r, 100));

    try {
        if (!camera) await initCamera();
        await camera.start();
        console.log("Camera started.");
        startBtn.style.display = 'none'; // Hide button after start
        statusText.textContent = "Camera Active | Model Ready";
        statusDot.style.background = '#00ff00';

        speak("System Ready", 'en'); // Audio feedback
    } catch (e) {
        console.error(e);
        startBtn.disabled = false;
        startBtn.textContent = "Retry Camera";
        statusText.textContent = "Camera Error";
        statusDot.style.background = 'red';
    }
});

loadModel();

