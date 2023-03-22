import torch
import os

from models import model_implements

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = 'cuda'
x = torch.rand([4, 3, 512, 512]).to(device)
model = model_implements.Swin_T(20, 3).to(device)
model.load_pretrained_imagenet('pretrained/swin_tiny_patch4_window7_224.pth', device)


with torch.no_grad():
    model.eval()
    model(x)

m = torch.jit.script(model)
torch.jit.save(m, 'Swin_Release.pt')
