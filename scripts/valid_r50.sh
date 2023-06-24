#!/usr/bin/env bash
echo "Running inference on" ${last_None_epoch_17_mean-iu_0.00000.pth}

     python -m torch.distributed.launch --nproc_per_node=1 valid.py \
        --val_dataset cityscapes bdd100k mapillary \
        --arch network.deepv3.DeepR50V3PlusD \
        --date 0302 \
        --bs_mult_val 1 \
        --exp r50_gtav \
        --snapshot ./logs/final/last_None_epoch_17_mean-iu_0.00000.pth