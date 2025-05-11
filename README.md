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

The Spacecraft Target Acquisition and Regression Datasets (STARDATA) have two variants being Near Range (denoted by NR) and Terminal Rnage (denoted by TR). Please refer to the paper (coming soon) for details about the dataset.

They are available through the following link: [STARDATA datasets](https://drive.google.com/drive/folders/1SKCJRe-ErgxVThGT8_DB7IGoR4GdR7Un?usp=drive_link).

### Clone Repository

Clone the repository using the following:

COPIABLE ADDRESS

In PoseEstimation. the pre-trained YOLO11s.pt model is avaible (trained on STARDATA-NR). Inside PoseEstimation/tools/, the following are available:

- Training script (dist_train.py).
- Validation script with pose metrics (Val_PoseMetrics.py).
- Live pose estimation script (PE_FullMission.py).

The pre-trained HigherHRNet (trained on STARDATA-TR) model can be found in PoseEstimation/output/coco_kpy/pose_higher_hrnet/w32_640_adam_lr1e-3/. This PoseEstimation folder is based off the COCO-based HigherHRNet strcuture and as a result retains the same structure and functionality.

In Orchestrator, the Main_ControlSystem_Tx.py script is the script that interfaces with the simulator and forms a closed-loop connection with teh receiver. It also hosts the 6DoF PID controller.

### Simulator

This work used Black Swan's Mission Design Simulator 0.8.3, which is availble through their [website](https://blackswanspace.com/mission-design-simulator/).

### CAD Model

The target spacercaft, which was used to geenarte the datasets, is a representation of the Tango satellite. It can be acquired from an [independent designer](https://www.cgtrader.com/3d-models/space/other/satellites-mango-and-tango).

The package comes with an accompanying satellite, but this can be deleted. Tango was then flipped 90 degrees about its X-axis and centered. As you can see in the code, the following transforms must be made before keypoint projection as a result of this and going from the Unity envrinoment to OpenCV:

- Axis Permutation: (1,2,0)
- Sign Flip: (−1, 1, 1)

The 3D keypoints for this specific model and camera intrinsics are available in PoseEstimation/tools/.

## Summary

These are all the fundamental parts of the ASTRIS framework. Once the simaultor is configured with Main_ControlSystem_Tx.py on an external PC which is connected to the ONX via Ethernet, PE_FullMission.py is executed from the ONX and opens a RTP conection. When the connection is accepted on the external PC, the simulator begins, the frames start being transmitetd, and the CV and control system begin operation.
