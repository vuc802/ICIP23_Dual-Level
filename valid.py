"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import numpy as np
import random
import cv2
from PIL import Image, ImageOps, ImageFilter, ImageFile

# Argument Parser
### set logging
logging.getLogger().setLevel(logging.INFO)

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepR50V3PlusD',
                    help='Network architecture. We have DeepR50V3PlusD (backbone: ResNet50) \
                    and DeepR101NV3PlusD (backbone: ResNet101).')
parser.add_argument('--dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia') # need to fix 9.16
parser.add_argument('--combine_all', action='store_true', default=False,
                    help='combine both train, val, and test sets of the source data')
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=['bdd100k'],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling, this one should not exceed 760 for synthia')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

## for DataParallel
# need to fix 9.16
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP') 
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')
####

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2') # need to understand 9.16

parser.add_argument('--trunk', type=str, default='resnet50',
                    help='trunk model, can be: resnet50 (default), resnet101') # fix model to 50 9.16
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=40000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=2,
                    help='Batch size for Validation per gpu') 
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=True)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--image_in', action='store_true', default=False,
                    help='Input Image Instance Norm')

parser.add_argument('--eval_epoch', type=int, default=1,
                    help='eval interval')

## style consistency
parser.add_argument('--sc_weight', type=float, default=10.0,
                    help='weight for consistency loss, e.g. js loss')

# ## imagenet cross entropy
parser.add_argument('--teacher', action='store_true', default=True,
                    help=' teacher: imgnet model')
parser.add_argument('--rc_layers', nargs='*', type=str, default=['layer0'],
                    help='a list of layers for retrospection loss : layer 0,1,2,3,4')
parser.add_argument('--rc_weights', nargs='*', type=float, default=[1.0],
                    help='weight for image net cross entropy')

parser.add_argument('--fd_layers', nargs='*', type=str, default=['layer4'],
                    help='a list of layers for retrospection loss : layer 0,1,2,3,4')
parser.add_argument('--fd_weights', nargs='*', type=float, default=[1.0],
                    help='weight for image net cross entropy')
parser.add_argument('--cos_weights', type=float, default=1.0,
                    help='weight for cosine similarity')

## wandb for logs
parser.add_argument('--wandb_name', type=str, default='',
                    help='use wandb and wandb name')

## new add
parser.add_argument('--medium_ch', type=int, default=3,
                    help='number of median channel')
parser.add_argument('--fd_half', action='store_true', default=False,
                    help=' no style constraint in FD')

args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
# args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

# torch.distributed.init_process_group(backend='nccl',
#                                      init_method=args.dist_url,
#                                      world_size=args.world_size,
#                                      rank=args.local_rank)

# palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
#            153, 153, 153, 250, 170, 30,
#            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
#            255, 0, 0, 0, 0, 142, 0, 0, 70,
#            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)
label_colours = [
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [  0,   0,   0]] # the color of ignored label(-1) 
label_colours = list(map(tuple, label_colours))

def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    prep_experiment(args, parser)
    writer = None

    _, _, _, extra_val_loaders = datasets.setup_loaders(args)

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args)

    # optim, scheduler = optimizer.get_optimizer(args, net)
    optim_net, scheduler_net = optimizer.get_optimizer(args, net)
    optim_ada, scheduler_adain = optimizer.get_optimizer(args, net)
    # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    # net = network.warp_network_in_dataparallel(net, args.local_rank)
    epoch = 0
    i = 0
    print("---")
    if args.snapshot:
        # epoch, mean_iu = optimizer.load_weights(net, optim, scheduler,
        #                                         args.snapshot, args.restore_optimizer)
        epoch, mean_iu = optimizer.load_weights(net, optim_net, optim_ada, scheduler_net, scheduler_adain,
                            args.snapshot, args.restore_optimizer)

    print("#### iteration", i)
    torch.cuda.empty_cache()
    # Main Loop
    # for epoch in range(args.start_epoch, args.max_epoch):

    for dataset, val_loader in extra_val_loaders.items():
        print("Extra validating... This won't save pth file")
        validate(val_loader, dataset, net, criterion_val, optim_net, scheduler_net, epoch, writer, i, save_pth=False)


def inv_preprocess(img, aug_type, number):

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)
        return img
                
    img = norm_ip(img, float(img.min()), float(img.max()))
    cv2.imwrite('./output/{}/{}.jpg'.format(aug_type, number), cv2.cvtColor(img.clone().cpu().numpy()*255., cv2.COLOR_RGB2BGR))
    


def decode_labels(mask, num_images=1, num_classes=19):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    h, w = mask.shape
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    for j_, j in enumerate(mask[:, :]):
        for k_, k in enumerate(j):
            if k < num_classes:
                outputs[j_, k_] = label_colours[k]
    outputs = np.array(outputs)
    return outputs#torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)

def validate(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []
    save_number=0

    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, gt_image, img_names, _ = data

        # assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        # assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            output = net(inputs, fixedAdain=True)
        output = output['main_out']


        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             datasets.num_classes)
        del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    # torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        mean_iu = evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

    return mean_iu


if __name__ == '__main__':
    main()