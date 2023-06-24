import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class class_dsu(nn.Module):
    """
    class level aumentation
    
    """
    def __init__(self, num_class):
        super(class_dsu, self).__init__()
        self.num_class = num_class
        self.eps = 1e-5

    def mask_generate(self, label, class_idx):
        # init
        inclass = False
        mask = torch.zeros_like(label)
        mask = torch.where(label==class_idx, 1., 0.)
        if torch.sum(mask, dim = [0,1,2]) > 0:
            inclass=True
        return inclass, mask
        
    def classnorm_std(self, x, mu, class_mask, ClassPixel_num):
        class_var = (((class_mask*(x - mu.reshape(x.shape[0], x.shape[1], 1, 1))).pow(2)).sum(dim=[2,3]))/ClassPixel_num
        class_std = (class_var + self.eps).sqrt()
        return class_std


    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def classuncertainty(self, x, class_mask):
        ClassPixel_num = class_mask.sum(dim = [2,3]) # Bx1
        a_ = torch.ones_like(ClassPixel_num).cuda()
        ClassPixel_num = torch.where(ClassPixel_num==0, a_, ClassPixel_num)
        wh_size = x.size()[2]*x.size()[3]
        mean = x.mean(dim=[2, 3], keepdim=False)*(wh_size)/ClassPixel_num # B,C / B,1
        std = self.classnorm_std(x, mean, class_mask, ClassPixel_num)

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)
        
        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)
        return x

    
    def forward(self, x, gt):  # red
        B, C, H, W = x.size() 
        class_x = torch.zeros_like(x).cuda()
        class_arr = np.arange(0, self.num_class)
        class_arr = np.append(class_arr, 255)
        
        for i in class_arr:
            inclass, class_mask = self.mask_generate(gt, i) # B, H, W
            if inclass:
                class_mask = F.interpolate(torch.unsqueeze(class_mask, 1), size=(H, W), mode='nearest')
                masked_x = x * class_mask
                masked_x = class_mask * self.classuncertainty(masked_x, class_mask)
                class_x += masked_x
        return class_x
