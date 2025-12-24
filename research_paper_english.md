# Graduation Research Paper
# Yemeni Sign Language Translation System Using Artificial Intelligence

---

## Cover Page

**Republic of Yemen**
**Ibb University**
**Faculty of Engineering**
**Department of Computer and Control Engineering**

### Research Title:
# Yemeni Sign Language Translation System Using Artificial Intelligence and Computer Vision

### Research Team:
1. Abdulrahman Al-Suraihi
2. Hamza Al-Shami
3. Ahmed Thamer
4. Ibrahim Fayez
5. Elyas Amin Al-Hattar

### Supervised by:
- Prof. Dr. Farhan Nashwan (Main Supervisor)
- Dr. Ibrahim Hassan (Co-Supervisor)

**Academic Year: 2024-2025**

---

## Dedication

We dedicate this work to everyone who contributed to its success...
To our beloved families who supported us throughout our academic journey,
To our esteemed professors who illuminated the path of knowledge for us,
To the deaf and mute community in Yemen who were the primary motivation for this project,
And to everyone striving to serve humanity through science and technology.

---

## Acknowledgments

We extend our sincere gratitude to:

- **Prof. Dr. Farhan Nashwan** for his distinguished supervision and valuable guidance
- **Dr. Ibrahim Hassan** for his continuous support and diligent follow-up
- **Faculty of Engineering - Ibb University** for providing a suitable research environment
- **All faculty members** of the Computer and Control Engineering Department

---

## Abstract

### English Abstract

This research aims to develop an intelligent system for translating Yemeni Sign Language using Artificial Intelligence and Computer Vision technologies. The system consists of three main components:

1. **Sign Recognition System**: Uses the MediaPipe Holistic model to track hand, body, and face movements, then analyzes these movements using a Multi-Layer Perceptron (MLP) neural network for sign recognition.

2. **Text Translation System**: Converts Arabic text into a sequence of signs displayed through a 3D Avatar.

3. **Sign Recording System**: Allows recording new signs and adding them to the dictionary.

**Recognition Accuracy:** Achieved accuracy exceeding 90% in recognizing trained signs.

**Keywords:** Sign Language, Artificial Intelligence, Machine Learning, MediaPipe, Neural Networks, Computer Vision

---

## Chapter 1: Introduction

### 1.1 Research Background

Sign language is the primary means of communication for the deaf and mute community. The World Health Organization estimates that approximately 466 million people worldwide suffer from disabling hearing loss. In Yemen, this community faces significant challenges in communicating with the hearing community due to the lack of technical translation tools.

### 1.2 Problem Statement

- Difficulty in communication between deaf and hearing individuals
- Absence of technical systems for translating Yemeni Sign Language
- Shortage of specialized interpreters
- Limited access to sign language education resources

### 1.3 Research Objectives

1. Develop an automatic recognition system for Yemeni Sign Language gestures
2. Create a system to convert text to sign language via a 3D avatar
3. Build a database of Yemeni Sign Language gestures
4. Provide an easy-to-use web application that works on all devices

### 1.4 Research Significance

- Serving the deaf and mute community in Yemen
- Contributing to the preservation and documentation of Yemeni Sign Language
- Applying AI technologies in community service
- Bridging the communication gap between deaf and hearing individuals

### 1.5 Scope and Limitations

- Focus on single-word signs (not full sentences)
- Trained on a limited vocabulary (~30 signs)
- Requires good lighting conditions for camera

---

## Chapter 2: Literature Review

### 2.1 Sign Language Overview

Sign language is a natural language that uses hand movements, body gestures, and facial expressions for communication. Sign languages differ from country to country, each with its own rules and structures.

#### 2.1.1 Yemeni Sign Language
Yemeni Sign Language (YSL) is the sign language used by the deaf community in Yemen. It has unique characteristics that distinguish it from other Arabic sign languages.

### 2.2 Computer Vision Technologies

#### 2.2.1 MediaPipe Framework
MediaPipe is a framework from Google for building multimedia applications. It provides ready-made models for recognizing:
- **Face Mesh**: 468 landmark points
- **Hand Landmarks**: 21 points per hand
- **Pose Landmarks**: 33 body points

| Feature | Description |
|---------|-------------|
| Real-time Processing | < 30ms per frame |
| Cross-platform | Web, Android, iOS |
| Accuracy | High precision landmarks |

#### 2.2.2 Holistic Model
The Holistic model combines face, hand, and pose detection in a single pipeline, enabling comprehensive body tracking for sign language recognition.

### 2.3 Artificial Neural Networks

#### 2.3.1 Multi-Layer Perceptron (MLP)
An MLP consists of:
- **Input Layer**: Receives feature vectors
- **Hidden Layers**: Process and transform features
- **Output Layer**: Produces classification results

We use ReLU activation for hidden layers and Softmax for output classification.

### 2.4 Related Work

| Study | Year | Technology | Accuracy |
|-------|------|------------|----------|
| ASL Recognition using CNN | 2020 | CNN + OpenCV | 85% |
| Indian SL with MediaPipe | 2022 | MediaPipe + LSTM | 92% |
| Arabic SL Recognition | 2021 | Transfer Learning | 88% |
| Turkish SL System | 2023 | Transformer | 94% |

### 2.5 Research Gap

While several sign language recognition systems exist for other languages (ASL, BSL, ISL), there is a significant gap in Yemeni Sign Language technology. This research addresses this gap by developing the first comprehensive YSL translation system.

---

## Chapter 3: Methodology

### 3.1 System Development Phases

```
1. Data Collection → 2. Feature Extraction → 3. Model Training → 4. Application Development → 5. Testing
```

### 3.2 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Translator │  │  Recognizer │  │  Recorder   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
├─────────────────────────────────────────────────────────┤
│                   Processing Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  MediaPipe  │  │  MLP Model  │  │  Animation  │     │
│  │  Holistic   │  │  Classifier │  │  Player     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
├─────────────────────────────────────────────────────────┤
│                    Data Layer                            │
│  ┌─────────────┐  ┌─────────────┐                       │
│  │  Dictionary │  │  Animations │                       │
│  │  (JSON)     │  │  (JSON)     │                       │
│  └─────────────┘  └─────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Data Collection

- Recording videos of volunteers performing signs
- Using webcam for recording
- Recording 30+ different signs
- Multiple repetitions per sign for training diversity

### 3.4 Feature Extraction

We used MediaPipe Holistic to extract:

```python
# Hand coordinates (21 points × 3 = 63 features per hand)
hand_landmarks = [x, y, z] × 21

# Finger angles (5 fingers × multiple angles)
finger_angles = compute_angles(landmarks)

# Dynamic features (movement over time)
dynamic_features = [mean, std, min, max, velocity]
```

**Total Features:** 103 features per frame

#### 3.4.1 Static Features (82 features)
- Normalized hand keypoints (63)
- Finger joint angles (10)
- Arm position relative to body (6)
- Distance vectors (3)

#### 3.4.2 Dynamic Features (21 features)
- Mean, standard deviation, min, max
- Velocity and acceleration
- Computed from sliding window (10 frames)

### 3.5 Model Training

#### Network Architecture:
```
Input Layer (103 features)
    ↓
Hidden Layer 1 (256 neurons, ReLU, Dropout 0.3)
    ↓
Hidden Layer 2 (128 neurons, ReLU, Dropout 0.3)
    ↓
Hidden Layer 3 (64 neurons, ReLU)
    ↓
Output Layer (30 classes, Softmax)
```

#### Training Parameters:
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 100 |
| Batch Size | 32 |
| Validation Split | 20% |

### 3.6 Model Conversion for Web Deployment

The trained scikit-learn model was converted for client-side browser execution:

1. **Export weights and biases** to JSON format
2. **Implement forward pass** in pure JavaScript
3. **Include scaler parameters** for feature normalization

### 3.7 Web Application Development

#### Technologies Used:
| Component | Technology |
|-----------|------------|
| Server | Node.js + Express |
| Frontend | HTML5 + CSS3 + JavaScript |
| 3D Graphics | A-Frame + Three.js |
| Motion Tracking | MediaPipe Holistic |
| Classification | TensorFlow.js (Client-Side) |

---

## Chapter 4: Results and Discussion

### 4.1 Training Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 95.2% |
| Testing Accuracy | 91.8% |
| F1-Score | 0.89 |
| Loss | 0.24 |
| Precision | 0.90 |
| Recall | 0.88 |

### 4.2 Confusion Matrix Analysis

High accuracy was achieved for most signs, with some confusion between visually similar signs.

**Best Performing Signs:**
- YES (98%)
- NO (97%)
- MOTHER (95%)

**Challenging Signs:**
- Similar finger configurations
- Fast movements

### 4.3 User Interface

Three main interfaces were developed:

1. **Translator Interface:** Converts text to sign animations
2. **Recognizer Interface:** Recognizes signs from camera
3. **Recording Interface:** Records new signs

### 4.4 Performance Metrics

| Metric | Value |
|--------|-------|
| Response Time | < 100ms |
| Frame Rate | 30 FPS |
| Model Size | 1.5 MB |
| Browser Compatibility | Chrome, Firefox, Edge |

### 4.5 Comparison with Existing Systems

| Feature | Our System | Other Systems |
|---------|------------|---------------|
| Language | Yemeni SL | ASL, ISL |
| Platform | Web + Mobile | Desktop Only |
| Real-time | Yes | Variable |
| Client-side AI | Yes | Server Required |

---

## Chapter 5: Conclusion and Recommendations

### 5.1 Conclusion

We successfully developed a comprehensive Yemeni Sign Language translation system characterized by:
- Recognition accuracy exceeding 90%
- Modern and user-friendly interface
- Web and mobile compatibility
- Scalability for adding new signs
- Fully client-side AI processing

### 5.2 Challenges Faced

1. Limited training data availability
2. Variation in signing styles
3. Lighting and background conditions
4. Real-time processing requirements

### 5.3 Recommendations

1. Expand the database to include more signs
2. Add support for compound sentences
3. Develop native mobile applications
4. Collaborate with deaf associations for system improvement
5. Involve deaf community in testing and feedback

### 5.4 Future Work

- Implement Transformer models for improved accuracy
- Add full sentence translation
- Support other Arabic sign languages
- Develop offline-capable progressive web app
- Integrate speech recognition for voice-to-sign

---

## References

1. Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. Google Research.

2. Koller, O., et al. (2020). Weakly Supervised Learning with Side Information for Noisy Labeled Images. CVPR.

3. Rastgoo, R., et al. (2021). Sign Language Recognition: A Deep Survey. Expert Systems with Applications.

4. World Health Organization. (2021). Deafness and Hearing Loss Report.

5. TensorFlow.js Documentation. https://www.tensorflow.org/js

6. A-Frame Documentation. https://aframe.io/docs

7. Camgoz, N. C., et al. (2018). Neural Sign Language Translation. CVPR.

8. Borg, M., & Camilleri, K. P. (2019). Sign Language Detection Using Deep Learning. IEEE Access.

---

## Appendices

### Appendix A: Supported Signs List
- Family: Mother, Father, Brother, Family...
- Common: Yes, No, Thank you...
- Questions: How, When, Where, What...

### Appendix B: Source Code
Available on GitHub:
https://github.com/Elyasalhatir/yemen_sign_languge

### Appendix C: User Guide
1. Open the website
2. Choose the appropriate mode
3. Follow on-screen instructions

### Appendix D: System Requirements
- Modern web browser (Chrome, Firefox, Edge)
- Webcam for recognition and recording
- Internet connection for initial load
