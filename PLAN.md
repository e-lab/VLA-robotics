# Robotics VLA Design Plan

This document outlines the design for a Vision-Language-Action (VLA) model for robotics, based on a modified **VLA-JEPA** architecture.

## 1. System Overview
The goal is to build a latent world model that processes multimodal sensor data (stereo vision, depth, proprioception) and natural language commands to predict a sequence of future actions and generated responses.

## 2. Input Specifications
The model processes two temporal **short sequences** to capture both instantaneous dynamics (optical flow) and long-term state changes.

### 2.1 Vision & Depth
- **Stereo Camera:** Two 640x480 RGB streams (Left/Right).
- **Depth Camera:** One 320x240 depth stream.
- **Temporal Window:** 
  - **Current Sequence:** $t, t-1, t-2$ (Captures high-frequency motion/flow).
  - **Past Sequence:** $t-N, t-N-1, t-N-2$ (Captures long-term intent and context).
- **Preprocessing & Fusion:** 
  - Depth is upsampled and aligned to RGB resolution.
  - **RGB-D Early Fusion:** For each camera (Left/Right), RGB and Depth are concatenated into a 4-channel tensor before patch embedding.
  - Total input: 6 frames of 4-channel data (3 current, 3 past) per eye.

### 2.2 Proprioception
- **State Vector:** 32-dimensional vector representing motor control outputs (joint positions/efforts).
- **Temporal Window:** Provided for all frames in the input window ($t, t-1, t-2$ and $t-N, t-N-1, t-N-2$).

### 2.3 Command
- **Command Vector:** 512-dimensional embedding representing the high-level instruction (e.g., encoded via CLIP or a similar text encoder).

## 3. Architecture: Modified VLA-JEPA
The architecture leverages the **Joint-Embedding Predictive Architecture (JEPA)** to learn a latent representation of the world dynamics.

### 3.1 Multimodal Encoder
- **RGB-D Vision Encoder (ViT):** A modified V-JEPA2 style encoder. 
  - Uses **3D Space-Time Patches** to process the short temporal sequences.
  - Patch embeddings fuse Left and Right stereo info along with Depth early in the network to learn spatial correspondence (disparity) and motion.
- **Proprioception & Command Projection:** Small MLP heads project the proprioception and command vectors into the same embedding dimension as the vision tokens.
- **Token Fusion:** All tokens (Vision, Depth, Proprioception, Command) are concatenated and passed through a Transformer backbone (e.g., Qwen3-VL-2B) to create a unified context representation $z_t$.

### 3.2 Latent Predictor (World Model)
- A JEPA-style predictor takes the context $z_t$ and a sequence of "action queries" to predict the future latent states $z_{t+1 \dots t+16}$.
- The predictor is conditioned on both the context $z_t$ and the **Command Vector**, allowing it to "imagine" outcomes based on intended goals.

### 3.3 Output Heads
- **Action Decoder:** A 16-step head (MLP or Diffusion-based) that decodes the predicted latent states into:
  - **32 Motor Control Outputs** (Target positions/efforts)
  - **32 Joint Velocities**
  - **32 Joint Speeds**
  - *Total: 96 values per timestep for 16 future steps.*
- **Response Decoder:** A separate head that decodes a 512-dimensional response vector (latent communication/high-level feedback).

## 4. Training Strategy
- **Self-Supervised Learning (JEPA):** Train the encoder and predictor to minimize the distance between predicted future latents and actual encoded future latents.
- **Temporal Consistency Loss:** A specialized loss to ensure that predicted **positions**, **velocities**, and **speeds** are mathematically consistent (i.e., velocity is the derivative of position).
- **Action Supervised Fine-tuning:** Use robotics datasets to map latents to specific motor actions.
- **Varying $N$:** The model is trained with randomized values of $N$ to ensure robustness across different timescales.

## 5. Development Roadmap
1. **Data Pipeline:** Implement loaders for stereo RGB-D sequences and proprioception vectors.
2. **Backbone Modification:** Adapt V-JEPA2 ViT for 4-channel (RGB-D) 3D patch embeddings.
3. **Consistency Loss:** Develop the mathematical constraints for the 96-dim action output.
4. **Predictor Implementation:** Build the temporal predictor for the 16-step horizon.
5. **Validation:** Test the latent representation's ability to reconstruct/predict future states and maintain motor consistency.
