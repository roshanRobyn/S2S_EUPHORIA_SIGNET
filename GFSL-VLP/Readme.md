# Why We Switched to This Architecture #
Initially, sign language recognition often relies on isolated "word-level" classification (like a visual dictionary). However, for a viable real-time product, users don't sign isolated words; they sign continuous sentences with co-articulation, fluid transitions, and complex grammar.

We switched to the Gloss-Free Sign Language Translation (GFSLT-VLP) architecture because it solves the "context" problem:

Continuous Translation: Instead of translating word-by-word, it translates full phrases.

Compositional Generalization: By leveraging mBART (a massive pre-trained language model by Facebook), the network already understands English grammar. This means if a user signs a novel mix of concepts the model hasn't explicitly seen together, mBART's internal logic can accurately decode the intent.

Skeleton-Driven Pipeline: To make this run in real-time on consumer hardware, we bypass heavy raw-pixel video processing. Instead, we use a custom keypoint_projector to feed pure skeletal math (extracted from tools like OpenPose/MediaPipe) directly into the language model.

# Current Status: Active Testing & Optimization #
This architecture is currently in the middle of active testing and integration.

Recent Milestones:

Successfully bypassed local VRAM hardware limits (4GB/8GB) by freezing mBART language weights, utilizing Automatic Mixed Precision (AMP), and implementing gradient clipping.

Engineered data-loader failsafes (Frame Guillotines for massive videos, padding for 1-frame glitches).

Validated end-to-end mathematical flow (Positive Loss generation and Checkpoint saving).

Next Engineering Steps:

The Extractor: Building the "Missing Link" script to map real-time MediaPipe Holistic output into the exact 411-float OpenPose coordinate format the network was trained on.

The Sliding Window: Transitioning from static video processing to a continuous rolling frame buffer for live webcam ingestion.

# Acknowledgements & Original Repository #
The foundational architecture and models modified in this subfolder are heavily based on the original GFSLT-VLP (Gloss-Free Sign Language Translation using Vision-Language Pretraining) research.

You can find the original paper and source code repository here:

Original GitHub Repository:https://github.com/zhoubenjia/GFSLT-VLP

Our modifications specifically focus on stripping down the heavy visual backbones, isolating the keypoint projection, and adapting the model for low-latency, real-time edge deployment.
