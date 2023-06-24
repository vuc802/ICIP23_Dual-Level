#!/usr/bin/env bash
    # Example on Cityscapes
     python -m torch.distributed.launch --nproc_per_node=1 train.py \
        --dataset gtav \
        --val_dataset cityscapes bdd100k mapillary \
        --arch network.deepv3.DeepR50V3PlusD \
        --city_mode 'train' \
        --lr_schedule poly \
        --lr 5e-4 \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 150000 \
        --bs_mult 2 \
        --gblur \
        --color_aug 0.5 \
        --date 0302 \
        --medium_ch 3 \
        --cos_weights 1.0 \
        --sc_weight 3.0 \
        --fd_layers 'layer3' 'layer4'\
        --fd_weights 1.5 1.5\
        --exp train_r50_Dual_Level \
        --ckpt ./logs/ \
        --tb_path ./logs/ \
        --wandb_name train_r50_Dual_Level\
        # --lr 0.000329 \
        # --snapshot ./logs/adain/last_None_epoch_8_mean-iu_0.00000.pth
 



