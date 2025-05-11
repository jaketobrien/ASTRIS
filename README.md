# ASTRIS - Autonomous Spacecraft Testing via Rendezvous, Imaging and Simulation

<p align="center">
  <img src="https://github.com/user-attachments/assets/8045ff3e-6ddb-491d-91a2-d8c149116c19" width="1000" alt="Pose_Pred_Vis">
</p>

ASTRIS is an open-source framework with open-source material to conduct simulated closed-loop autonomous spaceraft navigation with hardware-in-the-loop, model-in-the-loop, and sofware-in-the-loop.

The open-source material includes the following:

- Full software suite for operation.
- Two datasets.
- Pre-trained models.
- CAD model.
- Simulator.
- Set-up instructions.
- Hardware set-up instructions.

## Cite

If you find this work useful, please cite:

```bibtex
@misc{obrien2025astris,
  author       = {Jake O'Brien},
  title        = {ASTRIS: Autonomous Spacecraft Testing via Rendezvous, Imaging and Simulation},
  year         = {2025},
  howpublished = {\url{https://github.com/jaketobrien/ASTRIS}},
  note         = {Open-source framework for autonomous spacraft navigation simulation}
}
```

## Overview

This open-source framework combines a spacecraft CAD model, accompanying datasets, a tailored control system, and mission design simulation to validate a complete 6DoF pose estimation pipeline for uncooperative rendezvous scenarios. Existing studies tend to address individual components, such as pose estimation algorithms, synthetic dataset generation, control systems, or mission simulation, in isolation. In cases where integrated solutions do exist, they are typically closed-source and proprietary. This research bridges these gaps by presenting a unified, reproducible, and accessible framework, representing a significant advancement in the development and testing of Autonomous Spacecraft Navigation (ASN) systems. By making the full pipeline methodology open-source and releasing eligible components for public access, this work aims to accelerate research progress and promote accessibility for future innovation in the field. This open-source framework is named ASTRIS.

Within this 6DoF pose estimation pipeline evaluation, there are two networks; the Spacecraft Detection Network (SDN) and the Keypoint Regression Network (KRN). These are used depending on the phase of the simualted mission. This simulated mission has four phases with their relative distance forom the target spacraft:

- Far Range (>200 m)
- Near Range (200 m to 10 m)
- Terminal Range (10 m to 3 m)
- Docking (3 m to 0 m)

The focus is on the Near Range (NR) and Terminal Range (TR) in this framework.

### Near Range 

The NR Compter Vision (CV) pipeline for 6DoF pose estimation. Input images of size 1024x1024 pixels are passed to the SDN, which detects the target and crops the image to 640x640 from the bounding box center point. This cropped image is inputted to the KRN, which predicts 2D keypoints corresponding to the target spacecraft's key features. These keypoints are matched with a known 3D wireframe model of the target, and the pose is estimated using the EPnP algorithm with RANSAC for outlier rejection. A Kalman filter is then applied to refine the pose by smoothing temporal noise and ensuring consistency across frames. The final output is a 6D pose of the target.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4a08967b-0485-4ecb-b144-e5d68adc048f" width="1000" alt="NR_CV_Pipeline">
</p>

### Terminal Range

The TR CV pipeline for 6DoF pose estimation. This pipeline operates the exact same way as the NR CV pipeline, except it bypasses the SDN because the target is close enough to not need any detection or cropping.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a05b323b-ec30-46f5-a83e-fc31f56334ea" width="1000" alt="TR_CV_Pipeline">
</p>

## Setup

This section contains the instructions to setup and execute the system. The repository has been tested in a closed-loop system using a Nvidia Jetson Orin NX (ONX). The ONX was setup with Ubuntu 20.04 and JetPack 5.1.2.

This system leevrages the GPU capabilities of the ONX, with the insatllation guide available in ONX_Install.MD.

### Datasets

There are two datsets available:

- STARDATA-NR.
- STARDATA-TR.

The Spacecraft Target Acquisition and Regression Datasets (STARDATA) have two variants being Near Range (denoted by NR) and Terminal Rnage (denoted by TR). Please refer to the paper for details about the dataset.

They are availbale through the following link: https://drive.google.com/drive/folders/1SKCJRe-ErgxVThGT8_DB7IGoR4GdR7Un?usp=drive_link     PUT THIS LINK INTO THE WORDS STARDATA datasets.



### Clone Repository

Clone the repository using the following:

COPIABLE ADDRESS

In PoseEstimation. the pre-trained YOLO11s.pt model is avaible (trained on STARDATA-NR). Inside PoseEstimation/tools/, the follwoing are available:

MAKE BULLETPOINTS:
- Training script.
- Validation script with pose metrics.
- Live pose estimation script.

The pre-trained HigherHRNet (trained on STARDATA-TR) model can be found in PoseEstimation/output/... This folder is based off the COCO-based HigherHRNet strcuture adn as a result has the same structure and functionality.

In Orchestrator, the Main_ControlSystem_Tx.py script is the script that interfaces with the simulator and forms a closed-loop connection with teh receiver. It also hosts the 6DoF PID controller.

### Simulator

This work used Black Swan's Mission Design Simulator 0.8.2, which is availble through their website.

### CAD Model

The target spacercaft, which was used to geenarte the datasets, is a representation of the Tango satellite. It can be acquired from a {private designer} PUT LINK IN TWO WORDS.

The package comes with an accompanying satellite, but this can be deleted. Tango was then flipped 90 degrees about its X-axis and centered. As you can see in the code, the following transforms must be made before keypoint projection:

TRANSFORMS.

The 3D keypoints for this specific model are available in PoseEstimation/tools/.

DISCUSS SETUP FURTHER
