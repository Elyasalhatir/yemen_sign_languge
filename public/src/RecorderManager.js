export class RecorderManager {
    constructor(avatarController) {
        this.avatarController = avatarController;
        this.isRecording = false;
        this.recordedFrames = [];
        this.startTime = 0;

        this.setupUI();
    }

    setupUI() {
        this.recordBtn = document.getElementById('record-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.saveBtn = document.getElementById('save-btn');
        this.englishNameInput = document.getElementById('anim-name-en');
        this.arabicNameInput = document.getElementById('anim-name-ar');
        this.statusText = document.getElementById('recorder-status');

        if (this.recordBtn) this.recordBtn.addEventListener('click', () => this.startRecording());
        if (this.stopBtn) this.stopBtn.addEventListener('click', () => this.stopRecording());
        if (this.saveBtn) this.saveBtn.addEventListener('click', () => this.saveAnimation());

        this.updateButtons();
    }

    startRecording() {
        this.isRecording = true;
        this.recordedFrames = [];
        this.startTime = Date.now();
        if (this.statusText) this.statusText.innerText = "Recording...";
        this.updateButtons();
    }

    stopRecording() {
        this.isRecording = false;
        if (this.statusText) this.statusText.innerText = `Recorded ${this.recordedFrames.length} frames.`;
        this.updateButtons();
    }

    captureFrame(boneRotations) {
        if (!this.isRecording) return;

        const frame = {
            time: Date.now() - this.startTime,
            bones: JSON.parse(JSON.stringify(boneRotations))
        };
        this.recordedFrames.push(frame);
    }

    async saveAnimation() {
        const englishName = this.englishNameInput ? this.englishNameInput.value.trim() : '';
        const arabicName = this.arabicNameInput ? this.arabicNameInput.value.trim() : '';

        if (!englishName) {
            alert("Please enter the English name (filename).");
            return;
        }

        if (this.recordedFrames.length === 0) {
            alert("No frames recorded.");
            return;
        }

        try {
            const response = await fetch('/save-animation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    englishName,
                    arabicName,
                    frames: this.recordedFrames
                })
            });

            if (response.ok) {
                alert("Animation saved successfully!");
                this.recordedFrames = [];
                if (this.statusText) this.statusText.innerText = "Saved.";
                this.updateButtons();
            } else {
                alert("Error saving animation.");
            }
        } catch (e) {
            console.error(e);
            alert("Error saving animation.");
        }
    }

    updateButtons() {
        if (this.recordBtn) this.recordBtn.disabled = this.isRecording;
        if (this.stopBtn) this.stopBtn.disabled = !this.isRecording;
        if (this.saveBtn) this.saveBtn.disabled = this.isRecording || this.recordedFrames.length === 0;
    }
}
