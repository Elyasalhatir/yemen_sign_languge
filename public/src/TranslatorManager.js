export class TranslatorManager {
    constructor(avatarController) {
        this.avatarController = avatarController;
        this.isPlaying = false;
        this.dictionary = {};
        this.loadDictionary();

        this.setupUI();
    }

    async loadDictionary() {
        try {
            const response = await fetch('dictionary.json');
            if (response.ok) {
                this.dictionary = await response.json();
                this.populateDictionary(); // Render dictionary after loading
            }
        } catch (e) {
            console.warn("Could not load dictionary.json");
        }
    }

    populateDictionary() {
        this.dictionaryList = document.getElementById('dictionary-list');
        if (!this.dictionaryList) return;

        this.dictionaryList.innerHTML = '';

        // Sort keys alphabetically
        const sortedKeys = Object.keys(this.dictionary).sort((a, b) => a.localeCompare(b, 'ar'));

        let currentLetter = '';

        sortedKeys.forEach(key => {
            const value = this.dictionary[key];

            // Add Category Header (First Letter)
            const firstLetter = key.charAt(0).toUpperCase();
            if (firstLetter !== currentLetter) {
                currentLetter = firstLetter;
                const header = document.createElement('div');
                header.className = 'dict-header';
                header.innerText = currentLetter;
                // Style handled in CSS or inline here for safety
                header.style.cssText = 'padding: 8px 12px; background: rgba(0, 242, 234, 0.1); color: #00f2ea; font-weight: bold; margin-top: 15px; margin-bottom: 5px; border-radius: 8px; font-size: 1.1em; border-left: 3px solid #00f2ea;';
                this.dictionaryList.appendChild(header);
            }

            const item = document.createElement('div');
            item.className = 'dict-item';
            // Improved Item Styling
            item.style.cssText = 'display: flex; justify-content: space-between; align-items: center; padding: 10px 15px; margin-bottom: 8px; background: rgba(255, 255, 255, 0.05); border-radius: 8px; cursor: pointer; transition: all 0.2s ease;';
            item.onmouseover = () => { item.style.background = 'rgba(0, 242, 234, 0.15)'; item.style.transform = 'translateX(5px)'; };
            item.onmouseout = () => { item.style.background = 'rgba(255, 255, 255, 0.05)'; item.style.transform = 'translateX(0)'; };

            item.innerHTML = `
                <span class="dict-ar" style="font-weight: bold; font-size: 1.1em; color: #fff;">${key}</span>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="dict-arrow" style="color: #666;">→</span>
                    <span class="dict-en" style="color: #aaa; font-size: 0.9em;">${value}</span>
                </div>
            `;

            // Add click event
            item.onclick = () => {
                if (this.textInput) {
                    const currentText = this.textInput.value.trim();
                    this.textInput.value = currentText ? currentText + ' ' + key : key;
                    // Optional: Flash effect or sound to indicate addition
                    item.style.background = 'rgba(0, 255, 80, 0.2)';
                    setTimeout(() => { item.style.background = 'rgba(255, 255, 255, 0.05)'; }, 200);

                    // Auto-focus input
                    this.textInput.focus();
                }
            };

            this.dictionaryList.appendChild(item);
        });
    }

    setupUI() {
        this.textInput = document.getElementById('translator-input');
        this.playBtn = document.getElementById('play-btn');
        this.statusText = document.getElementById('translator-status');
        this.micBtn = document.getElementById('mic-btn');
        this.langRadios = document.getElementsByName('lang-choice');

        if (this.playBtn) this.playBtn.addEventListener('click', () => this.translateAndPlay());
        if (this.micBtn) this.micBtn.addEventListener('click', () => this.startVoiceInput());
    }

    startVoiceInput() {
        if (!('webkitSpeechRecognition' in window)) {
            alert("Voice input is not supported in this browser. Try Chrome.");
            return;
        }

        const recognition = new webkitSpeechRecognition();

        // Determine selected language for recognition
        let lang = 'ar-SA'; // Default
        let selectedLang = 'ar'; // Track which language was selected

        if (window.globalUI && window.globalUI.currentLang) {
            lang = window.globalUI.currentLang === 'ar' ? 'ar-SA' : 'en-US';
            console.log("Using Global Language:", lang);
        } else if (this.langRadios) {
            for (const radio of this.langRadios) {
                if (radio.checked) {
                    lang = radio.value === 'ar' ? 'ar-SA' : 'en-US';
                    break;
                }
            }
        }

        recognition.lang = lang;
        console.log("Voice recognition language set to:", lang); // Debug log

        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        if (this.statusText) this.statusText.innerText = "Listening (" + (lang === 'ar-SA' ? 'Arabic' : 'English') + ")...";
        if (this.micBtn) this.micBtn.classList.add('listening');

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            if (this.textInput) this.textInput.value = transcript;
            if (this.statusText) this.statusText.innerText = "Heard: " + transcript;
            console.log("Voice recognized:", transcript); // Debug log
            this.translateAndPlay(); // Auto-play
        };

        recognition.onerror = (event) => {
            console.error("Voice recognition error:", event.error);
            if (this.statusText) this.statusText.innerText = "Error: " + event.error;
            if (this.micBtn) this.micBtn.classList.remove('listening');
        };

        recognition.onend = () => {
            if (this.micBtn) this.micBtn.classList.remove('listening');
        };

        recognition.start();
    }

    async translateAndPlay() {
        const text = this.textInput.value.trim();
        if (!text) return;

        if (this.statusText) this.statusText.innerText = "جار ترجمة النص... | Translating...";

        try {
            // Split text into words
            const words = text.split(/\s+/);
            const animations = [];
            const foundWords = [];

            // Look up each word in the dictionary (supports Arabic OR English input)
            for (const word of words) {
                let animFile = null;
                const wordLower = word.toLowerCase();

                // 1. Try Arabic key exact match
                if (this.dictionary[word]) {
                    animFile = this.dictionary[word].toLowerCase();
                    foundWords.push(word);
                }
                // 2. Try Arabic key case-insensitive
                else {
                    const arabicMatch = Object.keys(this.dictionary).find(
                        key => key.toLowerCase() === wordLower || key.trim() === word.trim()
                    );
                    if (arabicMatch) {
                        animFile = this.dictionary[arabicMatch].toLowerCase();
                        foundWords.push(arabicMatch);
                    }
                    // 3. Try English value (reverse lookup)
                    else {
                        const englishMatch = Object.entries(this.dictionary).find(
                            ([ar, en]) => en.toLowerCase() === wordLower
                        );
                        if (englishMatch) {
                            animFile = englishMatch[1].toLowerCase();
                            foundWords.push(englishMatch[0]); // Show Arabic in status
                        } else {
                            console.log(`Word not in dictionary: ${word}`);
                        }
                    }
                }

                if (animFile) animations.push(animFile);
            }

            if (animations.length === 0) {
                if (this.statusText) this.statusText.innerText = "لم يتم العثور على كلمات في القاموس | No words found in dictionary";
                return;
            }

            if (this.statusText) this.statusText.innerText = `جار التشغيل: ${foundWords.join(' ')} | Playing...`;

            // Play animation sequence
            for (let i = 0; i < animations.length; i++) {
                const animName = animations[i];
                try {
                    await this.playAnimation(animName);
                    if (i < animations.length - 1) await this.pause(300);
                } catch (e) {
                    console.warn(`Animation not found: ${animName}`);
                }
            }

            await this.returnToRestPose();
            if (this.statusText) this.statusText.innerText = "تم | Done";

        } catch (e) {
            console.error("Translation error:", e);
            if (this.statusText) this.statusText.innerText = "خطأ | Error";
        }
    }


    // Helper function to pause between animations
    pause(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Return avatar to Rest Pose (A-Pose: Arms 45 degrees down)
    async returnToRestPose() {
        // Rest Pose: Arms relaxed down beside the body (A-Pose ~45 degrees down)
        // Quaternion for 45° rotation around Z-axis (PI/4)
        // sin(PI/8) = 0.38268, cos(PI/8) = 0.92388

        const restPoseData = {
            poseQuatArr: new Array(15).fill([0, 0, 0, 1]), // All bones to identity (neutral)
            leftHandQuatArr: new Array(21).fill([0, 0, 0, 1]), // Hands relaxed
            rightHandQuatArr: new Array(21).fill([0, 0, 0, 1])
        };

        // Set arms to A-Pose (45 degrees down)
        restPoseData.poseQuatArr[8] = [0, 0, 0.383, 0.924];   // Right arm (45° down)
        restPoseData.poseQuatArr[12] = [0, 0, -0.383, 0.924]; // Left arm (45° down)

        // Apply the pose
        this.avatarController.update(restPoseData);

        // Brief pause to show the Rest Pose
        await this.pause(500);
    }

    async playAnimation(word) {
        try {
            const response = await fetch(`/animations/${word}.json`);
            if (!response.ok) throw new Error("Not found");

            const frames = await response.json();
            await this.animateFrames(frames);
        } catch (e) {
            throw e;
        }
    }

    animateFrames(frames) {
        return new Promise((resolve) => {
            let startTime = null;

            // Detect frame format: Quaternion arrays (camera) vs Euler objects (manual)
            const isManualFormat = frames[0]?.bones && !frames[0].bones.poseQuatArr && typeof frames[0].bones === 'object';

            const animate = (timestamp) => {
                if (!startTime) startTime = timestamp;
                const progress = timestamp - startTime;

                // Find current frame based on timestamp
                let currentFrameIndex = 0;
                for (let i = 0; i < frames.length; i++) {
                    if (frames[i].time <= progress) {
                        currentFrameIndex = i;
                    } else {
                        break;
                    }
                }

                // Check if animation is complete
                if (currentFrameIndex >= frames.length - 1) {
                    // Apply last frame
                    const lastBones = frames[frames.length - 1].bones;
                    if (isManualFormat) {
                        this._applyManualBones(lastBones);
                    } else {
                        this.avatarController.update(lastBones);
                    }
                    resolve();
                    return;
                }

                const currentFrame = frames[currentFrameIndex];

                // Apply the frame
                if (isManualFormat) {
                    this._applyManualBones(currentFrame.bones);
                } else {
                    this.avatarController.update(currentFrame.bones);
                }

                requestAnimationFrame(animate);
            };

            requestAnimationFrame(animate);
        });
    }

    /**
     * Apply Euler bone rotations from manual studio recordings.
     * This directly manipulates the avatar's bone rotations using Three.js.
     */
    _applyManualBones(bones) {
        if (!this.avatarController || !this.avatarController.boneManager) return;

        const boneManager = this.avatarController.boneManager;

        Object.keys(bones).forEach(boneName => {
            const rot = bones[boneName];
            const bone = boneManager.getBone(boneName);
            if (bone) {
                if (rot.x !== undefined) bone.rotation.x = rot.x;
                if (rot.y !== undefined) bone.rotation.y = rot.y;
                if (rot.z !== undefined) bone.rotation.z = rot.z;
            }
        });
    }
}
