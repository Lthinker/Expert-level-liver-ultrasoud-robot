# Autonomous liver ultrasound robot
**Code for "Democratizing expert-level liver sonography through an autonomous lightweight ultrasound robot"**
📋 The complete project details have been submitted through the manuscript review system. Full project documentation and comprehensive codebase will be released following manuscript acceptance.

---
## Demo  
Here we provide a demo that can be run without a physical robot in `CodeForReviewDemo.zip`. The extraction password of the file is the manuscript ID (format with "nBME-xx-xxxx") which could be found from the peer review manuscript.  

The demo includes a detailed readme file with quick setup instructions to help you get started immediately.

## Core Components

### 🔥 Multimodal Fusion
- **File**: `diffusion_policy/zhutils/PosiImgFusion.py`
- **Key Method**: `AttnFusionKANForce.forward_Memory()`
- **Description**: Implements attention-based fusion of image, probe pose and force modalities and selects the top-k most important key frames

### 🎯 Pose Harmonization
- **Files**: 
  - `CodeForReview/diffusion_policy/policy/diffusion_zh.py`
  - `CodeForReview/communicate/envonline3cleanHSKE_XYXbot.py`
- **Key Methods**: 
  - `DiffusionUnetHybridImagePolicyForceContinousActionHistory.rel2abs()`
  - `USForceOnlineRead.cal_rel_pose()`
- **Description**: Computes the harmonized poses relative to the pose at the decision time point for generalization across devices and more efficient task understanding

### 🌟 Image Randomization
- **File**: `CodeForReview/DomainRam/DomainRam3Plus.py`
- **Key Class**: `FixedMaskConvexDomainRandomization`
- **Description**: Applies ultrasound-specific domain randomization techniques for generalization across subjects

### 🚀 Action Generation
- **File**: `CodeForReview/diffusion_policy/policy/diffusion_zh.py`
- **Key Method**: `DiffusionUnetHybridImagePolicyForceContinousActionHistory.predict_action_conduct()`
- **Description**: Generates robot actions (6D pose and 6D force) using diffusion model 

---

## Configuration

### Training Configuration
- **File**: `ExampleTrain.yaml`
- **Description**: Reference configuration file containing training hyperparameters and model settings

## Acknowledgments

This work builds upon the following excellent open-source projects:

- **Diffusion Policy**: [https://github.com/real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) - Foundational framework for diffusion-based policy learning
- **Efficient-KAN**: [https://github.com/Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan) - Efficient implementation of Kolmogorov-Arnold Networks

We gratefully acknowledge these contributions as the solid foundation for this project.

