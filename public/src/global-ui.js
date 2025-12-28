/**
 * Global UI Manager
 * Handles Theme Toggle, Language Toggle, and Camera Switching across all pages.
 */

class GlobalUIManager {
    constructor() {
        this.currentTheme = localStorage.getItem('theme') || 'dark';
        this.currentLang = localStorage.getItem('lang') || 'ar'; // Default to Arabic
        this.cameraFacingMode = 'user'; // 'user' (front) or 'environment' (back)

        this.init();
    }

    init() {
        // Apply saved preferences immediately
        this.applyTheme(this.currentTheme);

        // Wait for DOM to allow injecting UI
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupUI());
        } else {
            this.setupUI();
        }
    }

    setupUI() {
        this.injectControls();
        this.bindEvents();
        this.applyLanguage(this.currentLang); // Apply language text updates
    }

    injectControls() {
        // 1. Inject Floating Controls (Theme + Lang)
        const container = document.createElement('div');
        container.className = 'floating-controls';
        container.innerHTML = `
            <button class="ui-toggle-btn theme-toggle-btn" title="Toggle Theme (Light/Dark)"></button>
            <button class="ui-toggle-btn lang-toggle-btn" title="Toggle Language (Ar/En)">
                ${this.currentLang === 'ar' ? 'EN' : 'عربي'}
            </button>
        `;
        document.body.appendChild(container);

        // 2. Inject Camera Switch Button (if page has video element)
        const videoElement = document.getElementById('input_video');
        if (videoElement) {
            // Check if we are inside an a-scene or standard div
            const parent = videoElement.closest('.video-container') || videoElement.parentElement;
            if (parent) {
                const camBtn = document.createElement('button');
                camBtn.className = 'camera-switch-btn';
                camBtn.innerHTML = '📷';
                camBtn.title = 'Switch Camera';
                camBtn.onclick = () => this.switchCamera();

                // Position relative to parent
                if (parent.style.position !== 'absolute' && parent.style.position !== 'fixed') {
                    parent.style.position = 'relative';
                }
                parent.appendChild(camBtn);
            }
        }
    }

    bindEvents() {
        // Theme Toggle
        document.querySelector('.theme-toggle-btn').addEventListener('click', () => {
            this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
            this.applyTheme(this.currentTheme);
            localStorage.setItem('theme', this.currentTheme);
        });

        // Language Toggle
        document.querySelector('.lang-toggle-btn').addEventListener('click', (e) => {
            this.currentLang = this.currentLang === 'ar' ? 'en' : 'ar';
            e.target.innerText = this.currentLang === 'ar' ? 'EN' : 'عربي';
            this.applyLanguage(this.currentLang);
            localStorage.setItem('lang', this.currentLang);
        });
    }

    /* --- THEME LOGIC --- */
    applyTheme(theme) {
        if (!document.body) return; // Guard
        if (theme === 'light') {
            document.body.classList.add('light-mode');
        } else {
            document.body.classList.remove('light-mode');
        }

        // Broadcast event for 3D scenes (A-Frame)
        window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme } }));
    }

    /* --- LANGUAGE LOGIC --- */
    applyLanguage(lang) {
        const body = document.body;
        if (!body) return; // Guard for early calls

        // 2. Update all elements with data-ar / data-en attributes
        const TranslatableElements = document.querySelectorAll('[data-ar]');
        TranslatableElements.forEach(el => {
            const text = el.getAttribute(`data-${lang}`);
            if (text) {
                if (el.tagName === 'INPUT' && (el.type === 'text' || el.type === 'placeholder')) {
                    el.placeholder = text;
                } else {
                    el.innerText = text;
                }
            }
        });

        console.log(`Language applied: ${lang}`);
    }

    /* --- CAMERA LOGIC --- */
    async switchCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Camera not supported');
            return;
        }

        this.cameraFacingMode = this.cameraFacingMode === 'user' ? 'environment' : 'user';

        try {
            // Stop existing tracks
            const video = document.getElementById('input_video');
            if (video && video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
            }

            const constraints = {
                video: {
                    facingMode: this.cameraFacingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            if (video) {
                video.srcObject = stream;
                video.play();
            }

            console.log(`Camera switched to: ${this.cameraFacingMode}`);

        } catch (e) {
            console.error("Camera switch failed:", e);
            alert("Could not switch camera: " + e.message);
        }
    }
}

// Initialize Global UI
window.globalUI = new GlobalUIManager();
