import torch
import os
import shutil
from torch.utils.data import DataLoader

import torchvision
import dataset_medical




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
#sam = sam_model_registry["vit_h"](checkpoint='sam_vit_h_4b8939.pth')
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam = sam[0]
model = LoRA_Sam(sam,4).cuda()
path ="adla_no_multi.pth" 
model.load_state_dict(torch.load(path))


train_path = 'test'
cfg = dataset_medical.Config(datapath=train_path, savepath='./saved_model/msnet', mode='test', batch=16, lr=0.05, momen=0.9, decay=5e-4, epoch=50)
data = dataset_medical.Data(cfg)


test_loader = DataLoader(data, shuffle=False, batch_size=1)

outPath = 'test_no_multi'
if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.mkdir(outPath)

with torch.no_grad():
    model = model.eval()
    for i, (im1, _,_, label_name) in enumerate(test_loader):
        im1 = im1.cuda().float()
        label_name = label_name[0]
        #print(label_name)
        
        outputs = model(im1,1,512)
        outputs = outputs[0]
        a = outputs[1].unsqueeze(0)
     

        torchvision.utils.save_image(a, outPath + '/' +label_name+'.png')
