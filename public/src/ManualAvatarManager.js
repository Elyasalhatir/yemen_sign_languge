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
    }

    bindAvatar(avatar, type) {
        this.baseManager.bindAvatar(avatar, type || 'RPM');
        console.log("Premium Manager: Avatar Bound");
    }

    setAvatar(filename) {
        const avatarEl = document.getElementById('avatar') || document.getElementById('male');
        if (avatarEl) {
            avatarEl.setAttribute('src', 'avatars/' + filename);
        }
    }

    getBone(boneName) {
        // Advanced recursive lookup with naming variance
        let bone = null;

        // 1. Check Standard Managers (Pose/Hands)
        const managers = [
            this.avatarController.poseBoneManager,
            this.avatarController.leftHandBoneManager,
            this.avatarController.rightHandBoneManager
        ];

        for (let m of managers) {
            bone = m.getAvatarBone(boneName, 'RPM');
            if (bone) break;
        }

        // 2. Fallback: Deep Hierarchy Search (Handles Spine, Chest, and unconventional rigs)
        if (!bone) {
            const root = this.avatarController.poseBoneManager.getAvatarRoot();
            if (root) {
                root.traverse(child => {
                    if (child.isBone && !bone) {
                        const cleanName = child.name.replace(/^.*:/, '').toLowerCase(); // Strip prefixes like 'AvatarRoot:'
                        const searchName = boneName.toLowerCase();
                        if (cleanName === searchName || cleanName.endsWith(searchName)) {
                            bone = child;
                        }
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
            let mirrorValue = value;

            if (boneName.includes('Right')) mirrorBoneName = boneName.replace('Right', 'Left');
            else if (boneName.includes('Left')) mirrorBoneName = boneName.replace('Left', 'Right');

            if (mirrorBoneName) {
                // Symmetrical Inversion Logic
                if (axis === 'z' || axis === 'y') mirrorValue = value * -1;

                this._applyBoneRotation(mirrorBoneName, axis, mirrorValue);

                // Keep UI Sliders in Sync
                const slider = document.querySelector(`.bone-control[data-bone="${mirrorBoneName}"][data-axis="${axis}"]`);
                if (slider) {
                    slider.value = mirrorValue;
                    const badge = slider.parentElement.querySelector('.value-badge');
                    if (badge) badge.innerText = mirrorValue;
                }
            }
        }
    }

    /* --- Global Avatar Transforms --- */
    updateGlobal(type, axis, value) {
        const avatarEl = document.getElementById('avatar');
        if (!avatarEl) return;

        const floatVal = parseFloat(value);
        if (type === 'scale') {
            avatarEl.setAttribute('scale', `${floatVal} ${floatVal} ${floatVal}`);
        } else if (type === 'pos') {
            const currentPos = avatarEl.getAttribute('position');
            currentPos[axis] = floatVal;
            avatarEl.setAttribute('position', currentPos);
        } else if (type === 'rot') {
            const currentRot = avatarEl.getAttribute('rotation');
            currentRot[axis] = floatVal;
            avatarEl.setAttribute('rotation', currentRot);
        } else if (type === 'scale') {
            avatarEl.setAttribute('scale', `${floatVal} ${floatVal} ${floatVal}`);
        }

        // PERSISTENCE (V5)
        const settings = JSON.parse(localStorage.getItem('avatarSettings') || '{}');
        if (!settings[type]) settings[type] = {};
        if (type === 'scale') settings[type] = floatVal;
        else settings[type][axis] = floatVal;
        localStorage.setItem('avatarSettings', JSON.stringify(settings));
    }

    loadGlobalSettings() {
        const settings = JSON.parse(localStorage.getItem('avatarSettings') || '{}');
        const avatarEl = document.getElementById('avatar');
        if (!avatarEl || !settings) return;

        if (settings.scale) avatarEl.setAttribute('scale', `${settings.scale} ${settings.scale} ${settings.scale}`);
        if (settings.pos) {
            const p = avatarEl.getAttribute('position');
            if (settings.pos.x !== undefined) p.x = settings.pos.x;
            if (settings.pos.y !== undefined) p.y = settings.pos.y;
            if (settings.pos.z !== undefined) p.z = settings.pos.z;
            avatarEl.setAttribute('position', p);
        }
        if (settings.rot) {
            const r = avatarEl.getAttribute('rotation');
            if (settings.rot.y !== undefined) r.y = settings.rot.y;
            avatarEl.setAttribute('rotation', r);
        }

        // Sync UI Sliders
        document.querySelectorAll('.global-control').forEach(s => {
            const t = s.dataset.type;
            const a = s.dataset.axis;
            if (t === 'scale' && settings.scale) s.value = settings.scale;
            else if (t === 'pos' && settings.pos && settings.pos[a] !== undefined) s.value = settings.pos[a];
            else if (t === 'rot' && settings.rot && settings.rot[a] !== undefined) s.value = settings.rot[a];
        });
    }

    _applyBoneRotation(boneName, axis, value) {
        const bone = this.getBone(boneName);
        if (bone) {
            if (!this.boneState[boneName]) this.boneState[boneName] = { x: 0, y: 0, z: 0 };
            const floatVal = parseFloat(value);
            this.boneState[boneName][axis] = floatVal;
            bone.rotation[axis] = floatVal;
        }
    }

    updateFinger(handSide, fingerName, type, value) {
        value = parseFloat(value);
        this._applyFinger(handSide, fingerName, type, value);

        const isMirror = document.getElementById('mirror-mode')?.checked;
        if (isMirror) {
            const otherSide = handSide === 'Right' ? 'Left' : 'Right';
            let mirrorValue = value;

            // Invert spread (Y) and roll (Z) for mirror knuckles
            if (type === 'k_y' || type === 'k_z') mirrorValue = value * -1;

            this._applyFinger(otherSide, fingerName, type, mirrorValue);

            // Sync UI
            const slider = document.querySelector(`.finger-control[data-hand="${otherSide}"][data-finger="${fingerName}"][data-type="${type}"]`);
            if (slider) slider.value = mirrorValue;
        }
    }

    _applyFinger(handSide, fingerName, type, value) {
        if (type === 'curl') {
            // Joint Chain Curl
            // Thumb uses 0,1,2; Others use 1,2,3
            const joints = (fingerName === 'Thumb') ? [0, 1, 2] : [1, 2, 3];
            let axis = (fingerName === 'Thumb') ? 'y' : 'x';

            joints.forEach(i => {
                const bName = `${handSide}Hand${fingerName}${i}`;
                let dir = (handSide === 'Right' && fingerName === 'Thumb') ? -1 : 1;

                // Multipliers for natural closure
                let multiplier = 1.5;
                if (i === 1 || i === 0) multiplier = 1.1;
                if (i === 3 || i === 2) multiplier = 1.8;

                this._applyBoneRotation(bName, axis, value * multiplier * dir);
            });
        } else if (type.includes('_idx_')) {
            // Hyper-Granular Segment Control (e.g., k_x_idx_1)
            const parts = type.split('_'); // k, x, idx, 1
            const axis = parts[1];
            let index = parseInt(parts[3]);

            // MAPPING FIX: UI 1-2-3 maps to Bone 0-1-2 for Thumb, 1-2-3 for others
            if (fingerName === 'Thumb') index = index - 1;

            const bName = `${handSide}Hand${fingerName}${index}`;
            this._applyBoneRotation(bName, axis, value);
        } else if (type.startsWith('k_')) {
            // Legacy / Base Knuckle 3DOF (Joint 1)
            const axis = type.split('_')[1];
            const bName = `${handSide}Hand${fingerName}1`;
            this._applyBoneRotation(bName, axis, value);
        }
    }

    resetAll() {
        // Reset all active controlled bones to 0
        Object.keys(this.boneState).forEach(boneName => {
            const bone = this.getBone(boneName);
            if (bone) {
                bone.rotation.set(0, 0, 0);
                this.boneState[boneName] = { x: 0, y: 0, z: 0 };
            }
        });

        // Reset UI Sliders
        document.querySelectorAll('input[type="range"]').forEach(s => {
            s.value = 0;
            const badge = s.parentElement.querySelector('.value-badge');
            if (badge) badge.innerText = "0";
        });

        console.log("Premium Studio: All Resetted");
    }

    updateFist(handSide, value) {
        ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'].forEach(f => this.updateFinger(handSide, f, 'curl', value));
    }

    /* --- Preset Shortcuts --- */
    PRESETS = {
        // ✊ قبضة يمنى - Right Fist
        fist_right: () => {
            this.resetAll();
            ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'].forEach(f => {
                this._applyBoneRotation(`RightHand${f}1`, 'z', 1.4);
                this._applyBoneRotation(`RightHand${f}2`, 'z', 1.4);
                this._applyBoneRotation(`RightHand${f}3`, 'z', 1.2);
            });
        },
        // 🤛 قبضة يسرى - Left Fist  
        fist_left: () => {
            this.resetAll();
            ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'].forEach(f => {
                this._applyBoneRotation(`LeftHand${f}1`, 'z', -1.4);
                this._applyBoneRotation(`LeftHand${f}2`, 'z', -1.4);
                this._applyBoneRotation(`LeftHand${f}3`, 'z', -1.2);
            });
        },
        // 🖐️ كف مفتوح - Open Palm
        open_palm: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -0.5);
            this._applyBoneRotation('RightForeArm', 'z', 0.3);
        },
        // ☝️ إشارة - Point Up
        point_right: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -0.8);
            ['Middle', 'Ring', 'Pinky'].forEach(f => {
                this._applyBoneRotation(`RightHand${f}1`, 'z', 1.4);
                this._applyBoneRotation(`RightHand${f}2`, 'z', 1.4);
            });
            this._applyBoneRotation('RightHandThumb1', 'z', 0.8);
        },
        // 👍 إعجاب - Thumbs Up
        thumbs_up: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -0.3);
            this._applyBoneRotation('RightArm', 'x', 0.5);
            this._applyBoneRotation('RightForeArm', 'x', -1.5);
            ['Index', 'Middle', 'Ring', 'Pinky'].forEach(f => {
                this._applyBoneRotation(`RightHand${f}1`, 'z', 1.5);
                this._applyBoneRotation(`RightHand${f}2`, 'z', 1.5);
            });
        },
        // 👋 تلويح - Wave
        wave: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -1.8);
            this._applyBoneRotation('RightArm', 'x', 0.3);
            this._applyBoneRotation('RightForeArm', 'x', -0.4);
            this._applyBoneRotation('RightForeArm', 'z', 0.5);
        },
        // 🤟 أحبك - I Love You (ASL)
        ily: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -0.8);
            ['Middle', 'Ring'].forEach(f => {
                this._applyBoneRotation(`RightHand${f}1`, 'z', 1.5);
                this._applyBoneRotation(`RightHand${f}2`, 'z', 1.5);
            });
        },
        // 👌 موافق - OK Sign
        ok: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -0.5);
            this._applyBoneRotation('RightHandThumb1', 'z', 0.6);
            this._applyBoneRotation('RightHandThumb2', 'z', 0.4);
            this._applyBoneRotation('RightHandIndex1', 'z', 1.2);
            this._applyBoneRotation('RightHandIndex2', 'z', 1.0);
        },
        // 🙋 رفع اليد - Raise Hand
        arm_up_right: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -2.8);
            this._applyBoneRotation('RightArm', 'x', 0.2);
            this._applyBoneRotation('RightForeArm', 'x', 0.5);
        },
        // 🫀 يد على الصدر - Hand on Chest
        arm_chest: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -0.5);
            this._applyBoneRotation('RightArm', 'x', 0.8);
            this._applyBoneRotation('RightForeArm', 'x', -2.2);
            this._applyBoneRotation('RightForeArm', 'y', -0.8);
        },
        // ✌️ علامة النصر - Victory/Peace
        victory: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -0.6);
            ['Thumb', 'Ring', 'Pinky'].forEach(f => {
                this._applyBoneRotation(`RightHand${f}1`, 'z', 1.4);
                this._applyBoneRotation(`RightHand${f}2`, 'z', 1.4);
            });
            // Spread index and middle
            this._applyBoneRotation('RightHandIndex1', 'y', -0.3);
            this._applyBoneRotation('RightHandMiddle1', 'y', 0.3);
        },
        // 🤙 اتصل بي - Call Me
        call_me: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -0.5);
            this._applyBoneRotation('RightHand', 'z', -0.4);
            ['Index', 'Middle', 'Ring'].forEach(f => {
                this._applyBoneRotation(`RightHand${f}1`, 'z', 1.5);
                this._applyBoneRotation(`RightHand${f}2`, 'z', 1.5);
            });
        },
        // 🙏 دعاء - Prayer
        prayer: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -0.4);
            this._applyBoneRotation('RightArm', 'x', 1.2);
            this._applyBoneRotation('RightForeArm', 'x', -1.8);
            this._applyBoneRotation('LeftArm', 'z', 0.4);
            this._applyBoneRotation('LeftArm', 'x', 1.2);
            this._applyBoneRotation('LeftForeArm', 'x', -1.8);
        },
        // 💪 قوة - Strong Arm
        strong: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -2.2);
            this._applyBoneRotation('RightArm', 'x', 0.5);
            this._applyBoneRotation('RightForeArm', 'x', -2.5);
            ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'].forEach(f => {
                this._applyBoneRotation(`RightHand${f}1`, 'z', 1.4);
                this._applyBoneRotation(`RightHand${f}2`, 'z', 1.4);
            });
        },
        // 🤝 مصافحة - Handshake Ready
        handshake: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -0.8);
            this._applyBoneRotation('RightArm', 'x', 0.3);
            this._applyBoneRotation('RightForeArm', 'x', -0.8);
            this._applyBoneRotation('RightHand', 'z', -0.3);
        },
        // 🛑 توقف - Stop
        stop: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -1.5);
            this._applyBoneRotation('RightArm', 'x', 0.5);
            this._applyBoneRotation('RightForeArm', 'x', 0);
            this._applyBoneRotation('RightHand', 'x', 0.3);
        },
        // 👊 لكمة - Punch
        punch: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -1.2);
            this._applyBoneRotation('RightArm', 'x', 0.5);
            this._applyBoneRotation('RightForeArm', 'x', -0.3);
            ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'].forEach(f => {
                this._applyBoneRotation(`RightHand${f}1`, 'z', 1.5);
                this._applyBoneRotation(`RightHand${f}2`, 'z', 1.5);
            });
        },
        // 🖖 تحية فولكان - Vulcan Salute
        vulcan: () => {
            this.resetAll();
            this._applyBoneRotation('RightArm', 'z', -0.8);
            this._applyBoneRotation('RightHandIndex1', 'y', -0.4);
            this._applyBoneRotation('RightHandMiddle1', 'y', -0.2);
            this._applyBoneRotation('RightHandRing1', 'y', 0.2);
            this._applyBoneRotation('RightHandPinky1', 'y', 0.4);
        }
    };

    applyPreset(presetName) {
        if (this.PRESETS[presetName]) {
            this.PRESETS[presetName]();
            console.log(`Preset applied: ${presetName}`);
        }
    }

    /* --- Chain Controls (multi-joint) --- */
    updateArmChain(side, value) {
        // Moves shoulder and elbow together
        this.updateBone(`${side}Arm`, 'z', value);
        this.updateBone(`${side}ForeArm`, 'x', value * -0.5);
    }

    updateHandChain(side, value) {
        // Moves elbow and wrist together
        this.updateBone(`${side}ForeArm`, 'x', value);
        this.updateBone(`${side}Hand`, 'x', value * 0.3);
    }

    /* --- Timeline & Animation --- */
    startTime = null;

    captureFrame() {
        // Set startTime on first capture for relative timestamps
        if (this.startTime === null) this.startTime = Date.now();

        const frameTime = Date.now() - this.startTime;
        const frame = {
            time: frameTime,
            bones: JSON.parse(JSON.stringify(this.boneState))
        };
        this.frames.push(frame);
        this.renderTimeline();
    }

    deleteLastFrame() {
        if (this.frames.length > 0) {
            this.frames.pop();
            this.renderTimeline();
            if (this.frames.length > 0) this.restoreFrame(this.frames.length - 1);
        }
    }

    renderTimeline() {
        const container = document.getElementById('keyframe-container');
        if (!container) return;

        container.innerHTML = '';
        if (this.frames.length === 0) {
            container.innerHTML = '<div style="color: rgba(255,255,255,0.2); font-size: 0.8rem; padding: 10px;">لا يوجد لقطات حالياً...</div>';
            return;
        }

        this.frames.forEach((f, index) => {
            const node = document.createElement('div');
            node.className = 'keyframe-node';
            if (index === this.frames.length - 1) node.classList.add('active');
            node.innerHTML = `<strong>${index + 1}</strong><span>Frame</span>`;
            node.onclick = () => {
                document.querySelectorAll('.keyframe-node').forEach(n => n.classList.remove('active'));
                node.classList.add('active');
                this.restoreFrame(index);
            };
            container.appendChild(node);
        });
        container.scrollLeft = container.scrollWidth;
    }

    restoreFrame(index) {
        const frame = this.frames[index];
        if (!frame) return;

        // Support both old format (just bones) and new format ({ time, bones })
        const bones = frame.bones || frame;
        this.boneState = JSON.parse(JSON.stringify(bones));

        Object.keys(this.boneState).forEach(boneName => {
            const rot = this.boneState[boneName];
            const bone = this.getBone(boneName);
            if (bone) {
                if (rot.x !== undefined) bone.rotation.x = rot.x;
                if (rot.y !== undefined) bone.rotation.y = rot.y;
                if (rot.z !== undefined) bone.rotation.z = rot.z;
            }
        });
    }

    updateCombined(side, axis, value) {
        const val = parseFloat(value);
        ['Arm', 'ForeArm', 'Hand'].forEach(b => this.updateBone(side + b, axis, val));
        // Update UI
        ['Arm', 'ForeArm', 'Hand'].forEach(b => {
            const input = document.querySelector(`input[data-bone="${side + b}"][data-axis="${axis}"]`);
            if (input) input.value = val;
        });
    }

    updateArmElbow(side, axis, value) {
        const val = parseFloat(value);
        ['Arm', 'ForeArm'].forEach(b => this.updateBone(side + b, axis, val));
        // Update UI
        ['Arm', 'ForeArm'].forEach(b => {
            const input = document.querySelector(`input[data-bone="${side + b}"][data-axis="${axis}"]`);
            if (input) input.value = val;
        });
    }

    updateElbowWrist(side, axis, value) {
        const val = parseFloat(value);
        ['ForeArm', 'Hand'].forEach(b => this.updateBone(side + b, axis, val));
        // Update UI
        ['ForeArm', 'Hand'].forEach(b => {
            const input = document.querySelector(`input[data-bone="${side + b}"][data-axis="${axis}"]`);
            if (input) input.value = val;
        });
    }

    playAnimation() {
        if (this.frames.length < 1) return;
        if (this.isPlaying) return;

        this.isPlaying = true;
        document.getElementById('play-btn').innerText = '⏸';

        // Sort frames just in case
        this.frames.sort((a, b) => a.time - b.time);

        const startTime = Date.now();
        const lastFrameTime = this.frames[this.frames.length - 1].time;
        const loopDuration = lastFrameTime + 500; // 500ms pause at end

        const animate = () => {
            if (!this.isPlaying) return;

            const now = Date.now();
            const elapsed = (now - startTime) % loopDuration;

            // Find current interval
            let frameA = this.frames[0];
            let frameB = this.frames[this.frames.length - 1];

            // If elapsed is past last frame, hold last frame
            if (elapsed > lastFrameTime) {
                this.restoreFrame(this.frames.length - 1);
                this.highlightTimeline(this.frames.length - 1);
            } else {
                // Find frames to interpolate between
                for (let i = 0; i < this.frames.length - 1; i++) {
                    if (elapsed >= this.frames[i].time && elapsed < this.frames[i + 1].time) {
                        frameA = this.frames[i];
                        frameB = this.frames[i + 1];
                        break;
                    }
                }

                // Interpolate
                const range = frameB.time - frameA.time;
                const progress = range > 0 ? (elapsed - frameA.time) / range : 0;
                this.interpolateFrame(frameA, frameB, progress);

                // Highlight active keyframe in UI (approximate)
                const idx = this.frames.indexOf(frameA);
                this.highlightTimeline(idx);
            }

            this.playInterval = requestAnimationFrame(animate);
        };

        this.playInterval = requestAnimationFrame(animate);
    }

    interpolateFrame(frameA, frameB, t) {
        // Collect all unique bone keys from both frames
        const bonesA = frameA.bones || frameA; // Compat
        const bonesB = frameB.bones || frameB; // Compat
        const keys = new Set([...Object.keys(bonesA), ...Object.keys(bonesB)]);

        keys.forEach(boneName => {
            const rotA = bonesA[boneName] || { x: 0, y: 0, z: 0 };
            const rotB = bonesB[boneName] || { x: 0, y: 0, z: 0 };

            // Lerp Euler Angles
            const rx = this._lerp(rotA.x || 0, rotB.x || 0, t);
            const ry = this._lerp(rotA.y || 0, rotB.y || 0, t);
            const rz = this._lerp(rotA.z || 0, rotB.z || 0, t);

            // Apply directly to bone rotation for smoothness
            const bone = this.getBone(boneName);
            if (bone) {
                bone.rotation.x = rx;
                bone.rotation.y = ry;
                bone.rotation.z = rz;
            }
        });
    }

    _lerp(a, b, t) {
        return a + (b - a) * t;
    }

    highlightTimeline(index) {
        const nodes = document.querySelectorAll('.keyframe-node');
        nodes.forEach((n, i) => {
            if (i === index) n.classList.add('active');
            else n.classList.remove('active');
        });
    }

    stopAnimation() {
        if (this.playInterval) {
            clearInterval(this.playInterval);
            this.playInterval = null;
        }
        this.isPlaying = false;
        document.getElementById('play-btn').innerText = '▶';
    }

    async saveAnimation() {
        const nameEn = document.getElementById('anim-name-en').value.trim().toLowerCase().replace(/\s+/g, '_');
        const nameAr = document.getElementById('anim-name-ar').value.trim();

        if (!nameEn) { alert('الاسم الإنجليزي مطلوب | English Name Required'); return; }
        if (!nameAr) { alert('الاسم العربي مطلوب لإضافته للقاموس | Arabic Name Required for Dictionary'); return; }
        if (this.frames.length < 2) { alert('يجب التقاط أكثر من إطار | Capture at least 2 frames'); return; }

        // Convert frames to TranslatorManager-compatible format
        // TranslatorManager expects: { time, bones: { poseQuatArr, leftHandQuatArr, rightHandQuatArr } }
        // For manual studio, we store simpler Euler rotations; we'll convert to a compatible structure
        const formattedFrames = this.frames.map(f => ({
            time: f.time,
            bones: this._convertToQuatFormat(f.bones)
        }));

        try {
            const res = await fetch('/save-animation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    englishName: nameEn,
                    arabicName: nameAr,
                    frames: formattedFrames
                })
            });

            const data = await res.json().catch(() => ({}));

            if (res.ok) {
                alert(`✅ تم الحفظ بنجاح! | Saved as "${nameEn}.json"\nيمكنك استخدام "${nameAr}" في صفحة المترجم`);
                // Reset for next recording
                this.frames = [];
                this.startTime = null;
                this.renderTimeline();
                document.getElementById('anim-name-en').value = '';
                document.getElementById('anim-name-ar').value = '';
            } else if (res.status === 409) {
                // Duplicate error
                alert(`⚠️ ${data.message || 'This name already exists | هذا الاسم موجود مسبقاً'}`);
            } else {
                alert('❌ خطأ في الحفظ | Error Saving Animation');
            }
        } catch (e) {
            console.error(e);
            alert('❌ فشل الاتصال | Connection Failed');
        }
    }

    /**
     * Convert manual Euler bone state to TranslatorManager-compatible quaternion arrays.
     * NOTE: This is a simplified conversion. For more accurate animation, consider using Three.js Quaternion.
     */
    _convertToQuatFormat(bones) {
        // For compatibility, we'll store direct bone rotations in a format the translator can understand.
        // The translator's avatarController.update() expects: { poseQuatArr, leftHandQuatArr, rightHandQuatArr }
        // We'll create a simpler direct-update structure.
        return bones; // Keep as Euler for now; will update playback to handle both
    }
}

// Global Exports
const manualManager = new ManualAvatarManager();
window.manualManager = manualManager;

window.captureFrame = () => manualManager.captureFrame();
window.deleteLastFrame = () => manualManager.deleteLastFrame();
window.playAnimation = () => manualManager.playAnimation();
window.stopAnimation = () => manualManager.stopAnimation();
window.saveAnimation = () => manualManager.saveAnimation();
window.changeAvatar = (val) => {
    const el = document.getElementById('avatar');
    el.setAttribute('src', 'avatars/' + val);
};

// Event Binding
document.addEventListener('DOMContentLoaded', () => {
    const avatarEl = document.getElementById('avatar');
    avatarEl.addEventListener('model-loaded', () => {
        manualManager.bindAvatar(avatarEl.object3D.children[0]);
        manualManager.loadGlobalSettings(); // Apply persisted transforms (V5)

        // Fetch list
        fetch('/list-avatars').then(r => r.json()).then(avatars => {
            const sel = document.getElementById('avatar-selector');
            sel.innerHTML = '';
            avatars.forEach(a => {
                const opt = document.createElement('option');
                opt.value = opt.text = a;
                if (a === 'male.glb') opt.selected = true;
                sel.appendChild(opt);
            });
        });
    });

    // Delegates for inputs
    document.addEventListener('input', (e) => {
        const t = e.target;
        if (t.classList.contains('bone-control')) {
            manualManager.updateBone(t.dataset.bone, t.dataset.axis, t.value);
        } else if (t.classList.contains('finger-control')) {
            manualManager.updateFinger(t.dataset.hand, t.dataset.finger, t.dataset.type, t.value);
        } else if (t.classList.contains('global-control')) {
            manualManager.updateGlobal(t.dataset.type, t.dataset.axis, t.value);
        }
    });
});
