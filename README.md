# Dual-Level

To stabilize the training process, we train  
First, 
> image-level augmentation module (IAM): best mIoU may be roughly obtained at 8 epochs in GTAV  

Second, 
> class-level augmentation module (CAM): best mIoU may be roughly obtained at 17 epochs in GTAV

Therefore, please divide the learning process by  
epoch 1 ~ 8: train IAM <br>
epoch 9 ~ 20: train CAM <br>

We provide two pre-trained models, please load the path in args.snapshot if you want  
IAM  
https://drive.google.com/file/d/1OWc7ktIvycLniDRWON35NCNhQ_BlQfxV/view?usp=drive_link

CAM (final ver)
https://drive.google.com/file/d/1Xw9A7JUp-17WKHwINfJokormY6bgYzvX/view?usp=sharing
