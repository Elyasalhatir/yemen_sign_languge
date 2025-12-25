import MediapipeAvatarManager from './MediapipeAvatarManager.js';

class ManualAvatarManager {
    constructor() {
        this.baseManager = new MediapipeAvatarManager();
        this.avatarController = this.baseManager.avatarController;
        this.baseManager.setUseKalmanFilter(false);
        this.baseManager.setSlerpRatio(1.0);

        this.frames = [];
        this.isPlaying = false;
        this.playInterval = null;
        this.boneState = {};

        // Human Curl Ratios (Knuckle : Mid : Tip)
        this.CURL_RATIO = [1.0, 1.2, 0.8];
    }

    bindAvatar(avatar, type) {
        type = type || 'RPM';
        this.baseManager.bindAvatar(avatar, type);
        console.log("Avatar Bound for Universal Control:", type);
    }

    getBone(boneName) {
        // Precise bone lookup for standard rigs
        let bone = null;

        // 1. Check controllers
        if (boneName.includes('LeftHand') || boneName.includes('LeftArm')) {
            bone = this.avatarController.leftHandBoneManager.getAvatarBone(boneName, 'RPM');
        } else if (boneName.includes('RightHand') || boneName.includes('RightArm')) {
            bone = this.avatarController.rightHandBoneManager.getAvatarBone(boneName, 'RPM');
        } else {
            bone = this.avatarController.poseBoneManager.getAvatarBone(boneName, 'RPM');
        }

        // 2. Deep traverse fallback
        if (!bone) {
            const root = this.avatarController.poseBoneManager.getAvatarRoot();
            if (root) {
                root.traverse(child => {
                    if (child.isBone && !bone && (child.name === boneName || child.name.endsWith(boneName))) {
                        bone = child;
                    }
                });
            }
        }
        return bone;
    }

    updateBone(boneName, axis, value) {
        this._applyBoneRotation(boneName, axis, value);

        const isMirror = document.getElementById('mirror-mode')?.checked;
        if (isMirror) {
            let mirrorBoneName = null;
            if (boneName.includes('Right')) mirrorBoneName = boneName.replace('Right', 'Left');
            else if (boneName.includes('Left')) mirrorBoneName = boneName.replace('Left', 'Right');

            if (mirrorBoneName) {
                let mirrorValue = parseFloat(value);
                // Standard symmetry inversion for Y and Z
                if (axis === 'y' || axis === 'z') mirrorValue = -mirrorValue;

                this._applyBoneRotation(mirrorBoneName, axis, mirrorValue);

                // Sync slider if exists
                const slider = document.querySelector(`.bone-control[data-bone="${mirrorBoneName}"][data-axis="${axis}"]`);
                if (slider) slider.value = mirrorValue;
            }
        }
    }

    _applyBoneRotation(boneName, axis, value) {
        const bone = this.getBone(boneName);
        if (bone) {
            if (!this.boneState[boneName]) this.boneState[boneName] = { x: 0, y: 0, z: 0 };
            this.boneState[boneName][axis] = parseFloat(value);
            bone.rotation[axis] = this.boneState[boneName][axis];
        }
    }

    /**
     * Natural Finger Control
     * Applies rotation across all 3 joints in a human-like ratio
     */
    updateFingerCurl(handSide, fingerName, value) {
        value = parseFloat(value);

        // For RPM/Mixamo: 
        // Right Hand Curl: Negative X (usually)
        // Left Hand Curl: Negative X (usually)
        // Let's ensure 'Positive slider' = 'Close hand'
        const baseAngle = value * 1.5; // Up to ~90 degrees

        const joints = [1, 2, 3];
        joints.forEach((j, index) => {
            const bName = `${handSide}Hand${fingerName}${j}`;
            const ratio = this.CURL_RATIO[index];

            let axis = 'x';
            let polarity = -1; // Default curl is negative X for most joints

            // Special Thumb logic
            if (fingerName === 'Thumb') {
                axis = (j === 1) ? 'y' : 'x';
                if (handSide === 'Right') polarity = (j === 1) ? 1 : -1;
                else polarity = (j === 1) ? -1 : -1;
            }

            this._applyBoneRotation(bName, axis, baseAngle * ratio * polarity);
        });

        // Mirroring
        const isMirror = document.getElementById('mirror-mode')?.checked;
        if (isMirror) {
            const otherSide = handSide === 'Right' ? 'Left' : 'Right';
            this.updateFingerCurl(otherSide, fingerName, value); // Recursive but mirror-mode check prevents loop

            const slider = document.querySelector(`.finger-curl[data-hand="${otherSide}"][data-finger="${fingerName}"]`);
            if (slider) slider.value = value;
        }
    }

    updateFist(handSide, value) {
        const fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'];
        fingers.forEach(f => {
            // Internal call to avoid double mirroring logic
            this._applyNaturalCurl(handSide, f, value);
        });

        const isMirror = document.getElementById('mirror-mode')?.checked;
        if (isMirror) {
            const otherSide = handSide === 'Right' ? 'Left' : 'Right';
            fingers.forEach(f => this._applyNaturalCurl(otherSide, f, value));

            const slider = document.querySelector(`.fist-control[data-hand="${otherSide}"]`);
            if (slider) slider.value = value;
        }
    }

    _applyNaturalCurl(handSide, fingerName, value) {
        value = parseFloat(value);
        const baseAngle = value * 1.5;
        const joints = [1, 2, 3];
        joints.forEach((j, index) => {
            const bName = `${handSide}Hand${fingerName}${j}`;
            const ratio = this.CURL_RATIO[index];
            let axis = 'x';
            let polarity = -1;
            if (fingerName === 'Thumb') {
                axis = (j === 1) ? 'y' : 'x';
                polarity = (handSide === 'Right' && j === 1) ? 1 : -1;
            }
            this._applyBoneRotation(bName, axis, baseAngle * ratio * polarity);
        });
    }

    resetPose() {
        Object.keys(this.boneState).forEach(boneName => {
            const bone = this.getBone(boneName);
            if (bone) {
                bone.rotation.set(0, 0, 0);
            }
        });
        this.boneState = {};
        // Reset all sliders
        document.querySelectorAll('input[type="range"]').forEach(s => s.value = 0);
    }

    captureFrame() {
        const frame = JSON.parse(JSON.stringify(this.boneState));
        this.frames.push(frame);
    }

    deleteLastFrame() {
        if (this.frames.length > 0) {
            this.frames.pop();
            if (this.frames.length > 0) this.restoreFrame(this.frames.length - 1);
        }
    }

    restoreFrame(index) {
        const frame = this.frames[index];
        if (!frame) return;
        this.boneState = JSON.parse(JSON.stringify(frame));
        Object.keys(this.boneState).forEach(boneName => {
            const rot = this.boneState[boneName];
            const bone = this.getBone(boneName);
            if (bone) {
                bone.rotation.set(rot.x || 0, rot.y || 0, rot.z || 0);
            }
        });
    }

    playAnimation() {
        if (this.frames.length === 0) return;
        this.isPlaying = true;
        let i = 0;
        this.playInterval = setInterval(() => {
            if (i >= this.frames.length) i = 0;
            this.restoreFrame(i);
            i++;
        }, 200);
    }

    stopAnimation() {
        if (this.playInterval) clearInterval(this.playInterval);
        this.isPlaying = false;
    }
}

const manualManager = new ManualAvatarManager();
window.manualManager = manualManager;

// Global exports for UI
window.captureFrame = () => manualManager.captureFrame();
window.deleteLastFrame = () => { manualManager.deleteLastFrame(); updateTimelineUI(); };
window.playAnimation = () => manualManager.playAnimation();
window.stopAnimation = () => manualManager.stopAnimation();
window.resetPose = () => manualManager.resetPose();

function updateTimelineUI() {
    // Shared with HTML logic
    const container = document.getElementById('frame-trip');
    if (!container) return;
    container.innerHTML = '';
    if (manualManager.frames.length === 0) {
        container.innerHTML = '<span>Empty...</span>';
        return;
    }
    manualManager.frames.forEach((f, idx) => {
        const div = document.createElement('div');
        div.className = 'timeline-frame';
        div.innerText = idx + 1;
        div.onclick = () => manualManager.restoreFrame(idx);
        container.appendChild(div);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const avatarEl = document.getElementById('avatar');
    avatarEl.addEventListener('model-loaded', () => {
        manualManager.bindAvatar(avatarEl.object3D.children[0], 'RPM');
    });
});
