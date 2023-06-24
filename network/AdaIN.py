import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    """
    Image level augmetation
    """
    def __init__(self, args):
        super(AdaIN, self).__init__()
        self.args = args
        shape = (args.bs_mult, args.medium_ch)
        self.norm = nn.InstanceNorm2d(shape, affine=False) # output =(N,C,H,W) 
        self.alpha = nn.Parameter(torch.randn(shape)).cuda()
        self.alpha.data.normal_(0.,1.)
        self.beta = nn.Parameter(torch.randn(shape)).cuda()
        self.beta.data.normal_(0.,1.)
        

    def forward(self, x):
        img_mean = x.mean(dim=[2,3]) # B,C
        img_var = x.var(dim=[2,3]) # B,C
        img_sig = (img_var+1e-7).sqrt()
        ori_mean = img_mean.view(x.size(0), x.size(1), 1, 1)
        ori_sig = img_sig.view(x.size(0), x.size(1), 1, 1)
        alpha = self.alpha.view(x.size(0), x.size(1), 1, 1)
        beta = self.beta.view(x.size(0), x.size(1), 1, 1)

        return alpha * self.norm(x) + beta, torch.cat((ori_sig, ori_mean),dim=0), torch.cat((alpha, beta),dim=0)
