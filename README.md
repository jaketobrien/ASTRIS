# ASTRIS - Autonmous Spacecraft Testing via Rendezvous, Imaging and Simulation

<p align="center">
  <img src="https://github.com/user-attachments/assets/5d9ab63b-c2a0-4dde-9752-fd47cebc96dd" width="600" alt="Pose_Pred">
</p>


ASTRIS is an open-source methodology with open-source material to conduct simulated closed-loop autonomous spaceraft navigation with hardware-in-the-loop, model-in-the-loop, and sofware-in-the-loop.

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
  note         = {Open-source framework for spacecraft rendezvous simulation}
}```

## Summary

INFO AND IMAGE

## Setup

This section contains the instructions to execute the code. The repository has been tested in a closed-loop system using a Nvidia Jetson Orin NX. The ONX was setup with the following:

\bulletpoint: Jetpack X
......

This system leevrages the GPU capabilities of the ONX, with the insatllation guide available in ONX_Install.MD

### Datasets

There are two datsets available:

- STARDATA-NR.
- STARDATA-TR.

The Spacecraft Target Acquisition and Regression Datasets (STARDATA) have two variants being Near Range (denoted by NR) and Terminal Rnage (denoted by TR). Please refer to the paper for details about the dataset.

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
