import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
import pytorch_ssim
import pytorch_iou

import dataset_medical
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic

from sam_lora_image_encoder import LoRA_Sam

sam = sam_model_registry["vit_b"](checkpoint='sam_vit_b_01ec64.pth')#"sam_vit_b_01ec64.pth")
sam = sam[0]
model = LoRA_Sam(sam,4).cuda()

#pretrain = 'sam_vit_h_4b8939.pth'
pretrain ="sam_vit_b_01ec64.pth" 
model.load_lora_parameters(pretrain)
#path ="samed_.pth" 
#model.load_state_dict(torch.load(path))

train_path = 'train'
cfg = dataset_medical.Config(datapath=train_path, savepath='./saved_model/msnet', mode='train', batch=16, lr=0.05, momen=0.9, decay=5e-4, epoch=50)
data = dataset_medical.Data(cfg)


warnings.filterwarnings("ignore")
ssim_loss = pytorch_ssim.SSIM(window_size=7,size_average=True).cuda()
iou_loss = pytorch_iou.IOU().cuda()

model = model.train()
ce_loss = nn.CrossEntropyLoss()
deal = nn.Softmax(dim=1)
base_lr = 0.005
EPOCH = 50
LR= 0.01

warmup_period  = 2950
print(warmup_period)
b_ = base_lr/warmup_period 

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, betas=(0.9, 0.999), weight_decay=0.1)





train_loader= DataLoader(data,
                      shuffle=True,
                      batch_size=8,
                      pin_memory=True,
                      num_workers=16,
                      )


losses0 = 0
losses1 = 0
losses2 = 0
losses3 = 0
losses4 = 0
losses5 = 0
print(len(train_loader))

def adjust_learning_rate(optimizer,epoch,start_lr):
    if epoch%22 == 0:  #epoch != 0 and 
    #lr = start_lr*(1-epoch/EPOCH)
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"]*0.1
        print(param_group["lr"])
        
        
iter_num = 0 
LR=0.01
max_iterations = 29500
for epoch_num in range(EPOCH):
    print(epoch_num)
    adjust_learning_rate(optimizer,epoch_num,LR)
   
    print('LR is:',optimizer.state_dict()['param_groups'][0]['lr'])
    show_dict = {'epoch':epoch_num}
    for i_batch,(im1,label0) in enumerate(tqdm.tqdm(train_loader,ncols=60,postfix=show_dict)):  #,edge0,edge1,edge2,edge3
        im1 = im1.cuda().float()
        label0 = label0.cuda()
        

        
        outputs = model(im1,1,512)#[:,:2,:,:]
#        print(outputs.size())
#         outputs = outputs[0][1].unsqueeze(0)
#         torchvision.utils.save_image(outputs, '1.png')

#         torchvision.utils.save_image(label0.unsqueeze(0), 'true.png')
#         break
        
        loss0 = ce_loss(outputs[0],label0.long())+(1-ssim_loss(deal(outputs[0]),label0))+iou_loss(deal(outputs[0]),label0)
        loss1 = ce_loss(outputs[1],label0.long())+(1-ssim_loss(deal(outputs[1]),label0))+iou_loss(deal(outputs[1]),label0)
        loss2 = ce_loss(outputs[2],label0.long())+(1-ssim_loss(deal(outputs[2]),label0))+iou_loss(deal(outputs[2]),label0)
        loss3 = ce_loss(outputs[3],label0.long())+(1-ssim_loss(deal(outputs[3]),label0))+iou_loss(deal(outputs[3]),label0)
        loss4 = ce_loss(outputs[4],label0.long())+(1-ssim_loss(deal(outputs[4]),label0))+iou_loss(deal(outputs[4]),label0)
       
#+(1-ssim_loss(deal(outputs),label0))+iou_loss(deal(outputs),label0)
#         loss1 = all_loss(outputs[1],label1)##+(1-ssim_loss(deal(outputs[1]),label1))+iou_loss(deal(outputs[1]),label1)
#         loss2 = all_loss(outputs[2],label2)#+(1-ssim_loss(deal(outputs[2]),label2))+iou_loss(deal(outputs[2]),label2)
#         loss3 = all_loss(outputs[3],label3)#+(1-ssim_loss(deal(outputs[3]),label3))+iou_loss(deal(outputs[3]),label3)

       
        loss = loss0+loss1+loss2+loss3+loss4#+0.05*loss5
       
        
        losses0 += loss0
        losses1 += loss1
        losses2 += loss2
        losses3 += loss3
        losses4 += loss4
        #losses5 += 0.05*loss5
        
        
        optimizer.zero_grad()
        #scheduler(optimizer,i_batch,epoch_num)
        loss.backward()
        optimizer.step()
        '''
        if iter_num < warmup_period:
            lr_ = base_lr * ((iter_num + 1) / warmup_period)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
        else:
            shift_iter = iter_num - warmup_period
            lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
                    
        iter_num = iter_num + 1
        '''
        if i_batch%50 == 0:
            print(i_batch,'|','losses0: {:.3f}'.format(losses0.data),'|','losses1: {:.3f}'.format(losses1.data),'|','losses2: {:.3f}'.format(losses2.data),'|','losses3: {:.3f}'.format(losses3.data),'|','losses4: {:.3f}'.format(losses4.data))
            #,'|','losses1: {:.3f}'.format(losses1.data),'|','losses2: {:.3f}'.format(losses2.data),'|','losses3: {:.3f}'.format(losses3.data)

            
            losses0=0
            losses1=0
            losses2=0
            losses3=0
            losses4=0
       
    torch.save(model.state_dict(),'adla_no_multi.pth')
