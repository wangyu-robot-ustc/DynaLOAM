# [Autonomous Robots 2025] DynaLOAM: Robust LiDAR Odometry and Mapping in Dynamic Environments.
## Affiliation: Networked Robotics and Systems Lab, HITSZ

## Introduction 
Simultaneous localization and mapping (SLAM) based on LiDAR in dynamic environments remains a challenging problem due to unreliable data association and residual ghost tracks in the map. In recent years, some related works have attempted to utilize semantic information or geometric constraints between consecutive frames to reject dynamic objects as outliers. However, challenges persist, including poor real-time performance, heavy reliance on meticulously annotated datasets, and susceptibility to misclassifying static points as dynamic. This paper presents a novel dynamic LiDAR SLAM framework called DynaLOAM, in which a complementary dynamic interference suppression scheme is exploited. For accurate relative pose estimation, a lightweight detector is proposed to rapidly respond to pre-defined dynamic object classes in the LiDAR FOV and eliminate correspondences from dynamic landmarks. Then, an online submap cleaning method based on visibility and clustering is proposed for real-time dynamic object removal in submap, which is further utilized for pose optimization and global static map construction. By integrating the complementary characteristics of prior appearance detection and online visibility check, DynaLOAM can finally achieve accurate pose estimation and static map construction in dynamic environments. Extensive experiments are conducted on the KITTI dataset and three real scenarios. The results show that our approach achieves promising performance compared to state-of-the-art methods.

## Usage
### Object detection
Refer to the README in RobDet3D to compile the model engine.
Run src/RobDet3D/tools/deploy/pc_det_thread.py for detection. You need to modify the paths of sharelib and model in it.
### LiDAR Odometry
