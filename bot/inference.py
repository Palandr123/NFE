import sys

from StyleGAN import *


def inference():
    img_size = 64
    step = int(math.log(img_size, 2)) - 2
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    G = torch.load("Generator_v2_150.pth", map_location=device)
    z = torch.randn((1, 512))
    with torch.no_grad():
        img = G(z, step=step)[0]
        return img


if name == 'main':
    inference()
