# Single-Domain Generalization for Semantic Segmentation via Dual-Level Domain Augmentation (Dual-Level)
The paper has been accepted for presentation at IEEE ICIP 2023.

## Overview
In this paper, we propose a dual-level augmentation method to explicitly enrich the domain diversity from the perspective of image- and class-level style augmentation. First, to enrich the image-level domain diversity, we propose an Image-Level Augmentation Module (IAM) via the learnable but doubly-constrained AdaIN. Second, to enlarge per-class styles, we propose a Class-Level Augmentation Module (CAM) to dynamically adapt the class-level style in terms of the per-class batch statistics. Next, we propose a domain-generalized feature learning and leverage a pre-trained image-classification model with a two-fold goal: to learn feature representation which generalizes well to both the source and the augmented domains, and to enforce the features compliant with the distilled semantics from the pre-trained model.


## Training
To stabilize the training process, please divide the learning process into  
epoch 1 ~ 8: train IAM <br>
epoch 9 ~ 20: train CAM <br>

We provide two pre-trained models, please load the path in args.snapshot  
IAM  https://drive.google.com/file/d/1OWc7ktIvycLniDRWON35NCNhQ_BlQfxV/view?usp=drive_link

CAM (can obtain the final result and qualitative visualization in the paper)
https://drive.google.com/file/d/1Xw9A7JUp-17WKHwINfJokormY6bgYzvX/view?usp=sharing
