# Dual-Level

To stabilize the training process, we train  
First, 
> image-level augmentation module (IAM): best mIoU may be roughly obtained at 8 epochs in GTAV  

Second, 
> class-level augmentation module (CAM): best mIoU may be roughly obtained at 17 epochs in GTAV

Therefore, please divide the learning process by  
epoch 1~8: train IAM

epoch 9~20: train CAM


 
