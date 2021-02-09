import torch.hub

from _models import SVG_FROZEN

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
