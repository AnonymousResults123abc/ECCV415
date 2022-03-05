import torch
from collections import OrderedDict

new = OrderedDict()
for k,v in torch.load('66_7_new.pth',map_location='cpu').items():
    if 'alpha' in k:
        new[k.replace('alpha', 'scale')] = v
    else:
        new[k] = v
        
torch.save(new, 'rbonn_66_7.pth')