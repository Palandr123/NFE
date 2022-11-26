import sys

sys.path.append(r"D:\Education\Innopolis\s7\pml\Project\NFE")
from StyleGAN import *
from manipulate import manipulate
import matplotlib.pyplot as plt
import numpy as np
from telebot import types


def generate_and_save_random_image(G):
    img_size = 64
    step = int(math.log(img_size, 2)) - 2
    z = torch.randn((1, 512))
    with torch.no_grad():
        img = G(z, step=step)[0]
    imgpath = 'random_image.png'
    imgdata = torch.clip(img, 0, 1).permute([1, 2, 0]).detach().cpu().numpy()
    plt.imsave(imgpath, imgdata)
    return imgpath, z


def change_image(G, z, feature='hair', value='green', preserved_features=None):
    img_size = 64
    z_s = manipulate(z, feature, value, preserved_features, start=-5.0, end=5.0)
    step = int(math.log(img_size, 2)) - 2
    pics = []
    filepath = 'fig{}.png'
    for i, z in enumerate(z_s):
        out = G(torch.tensor(z).view(1, -1).float(), step=step)
        img = out[0].permute([1, 2, 0]).detach().cpu().numpy()
        img = np.clip(img, 0, 1)
        plt.imsave(filepath.format(i), img)
        pics.append(open(filepath.format(i), 'rb'))
    return pics


if __name__ == '__main__':
    # inference()
    pass