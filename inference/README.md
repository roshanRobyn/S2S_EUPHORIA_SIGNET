# 🗣️ Sign2Sound Euphoria

> **Bridging the gap between Sign Language and Spoken English with Real-Time, Edge-Computed AI.**

[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-blue?style=for-the-badge&logo=google)](https://google.github.io/mediapipe/)
[![Platform](https://img.shields.io/badge/Hardware-RTX%203050%20(4GB)-green?style=for-the-badge&logo=nvidia)](https://www.nvidia.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## 📖 Overview
**Sign2Sound Euphoria** is a bi-directional Sign Language Translation system designed to run entirely on consumer-grade hardware (Offline-First). It eliminates the need for expensive cloud APIs or heavy server-grade GPUs.

By utilizing a novel **Dual-Expert Graph Neural Network (ST-GCN)** architecture, the system distinguishes between **Dynamic Words** (WLASL) and **Static Finger-Spelling** (ASL) in real-time. It integrates a **Small Language Model (SLM)** to correct raw glosses into grammatically natural English sentences.

### 🚀 Key Innovations
* **Dual-Expert Routing:** Separate specialized models for Spelling vs. Signing to eliminate the "Hold vs. Letter" confusion.
* **Edge-Optimized:** Runs at **22+ FPS** on a laptop RTX 3050 (4GB VRAM).
* **Hybrid Pipeline:** Combines Vision (ST-GCN) + Language (SLM) for context-aware translation.
* **Privacy First:** Zero data leaves the device; fully offline execution.

---

## 🛠️ System Architecture

The pipeline processes video input in four distinct stages:

1.  **Skeletal Extraction:**
    * **Tool:** Google MediaPipe Holistic.
    * **Data:** Extracts **109 Keypoints** (Body, Hands, Face) per frame.
    * **Normalization:** Relative Nose-Centric Alignment (invariant to user position).

2.  **Dual-Expert Inference (ST-GCN):**
    * **Expert A (WLASL):** Tracks temporal motion for dynamic words (e.g., "Mother", "Eat").
    * **Expert B (ASL):** Recognizes static spatial features for finger-spelling (e.g., "A-D-A-M").

3.  **Grammar Correction (SLM):**
    * **Input:** Raw Glosses (e.g., *"Who Eat Now"*).
    * **Model:** Quantized Microsoft Phi-2 / DistilGPT-2.
    * **Output:** Natural English (e.g., *"Who is eating now?"*).

4.  **Vocalization (Coming Soon):**
    * **Engine:** KokoroTTS (High-fidelity, <80ms latency).

---

## 📊 Performance Metrics

We evaluated the system on a held-out test set (20% split) using an ASUS TUF A15 (RTX 3050).

| Dataset / Task | Accuracy | F1-Score | Latency |
| :--- | :---: | :---: | :---: |
| **ASL Letters (Static)** | **99.04%** | 0.99 | 45ms |
| **WLASL-100 (Dynamic)** | **92.05%** | 0.91 | 45ms |
| **End-to-End Pipeline** | N/A | N/A | ~22 FPS |

> *Note: Training graphs and confusion matrices are available in the `results/` directory.*

---

## 📦 Installation

### Prerequisites
* Python 3.10+
* NVIDIA GPU (Recommended) or CPU
* Webcam

### Setup
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/Sign2Sound-Euphoria.git](https://github.com/yourusername/Sign2Sound-Euphoria.git)
    cd Sign2Sound-Euphoria
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Models**
    * Place `stgcn_wlasl100_final.pth` in `models/`.
    * Place `stgcn_letters_scratch.pth` in `models/`.
    *(Pre-trained weights link to come)*

---

## 💻 Usage / Demos

### The "Final Pipeline" Demo (Words + Grammar)
Runs the full stack: Video -> Gloss -> SLM Correction.

```bash
python inference/final_pipeline_demo.py
```
* **Input:** Sequence of videos (e.g., `who.mp4`, `eat.mp4`, `now.mp4`).
* **Output:** `[SLM]: Who is eating now?`

## 📂 Dataset Information
We utilized a **Split-Dataset Strategy** to solve class imbalance and confusion:

1. **IEEE DataPort ASL Dataset:** Used for training the static Spelling Expert (Filtered to ~200 samples/class).
2. **WLASL (World Level ASL):** Used top 100 classes for the Dynamic Word Expert.

> **Access:** Dataset composition details available [here](https://tinyurl.com/Sign2Sound).

---

## 🔮 Future Roadmap
- [ ] **KokoroTTS Integration:** Replace text output with natural voice synthesis.
- [ ] **Streaming Decoder:** Optimize SLM to decode tokens asynchronously for lower latency.
- [ ] **Mobile Port:** Quantize models for deployment on Android/iOS via TFLite.

---

## 👥 Team
* **Roshan Robin** - AI Engineer & Architecture
* **Jayalakshmy Jayakrishnan** - Data Processing & Evaluation
* **Nima Fathima** - Frontend & Integration
* **Sakhil N Maju** - Frontend & Integration
---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
