import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_base_image():
    base_image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './Map_v2.png')
    base_image = cv2.imread(base_image_path)
    return base_image


def renewal_map(img, observation):
    Pacman_left = (observation[10] - 5, observation[16]+3)
    Pacman_right = (observation[10] - 11, observation[16]+10)
    cv2.rectangle(img, Pacman_left, Pacman_right, (0,0,0), -1)
    return img


def pacman_obs_Ram(observation):
    parameter = {
        'enemies': [
            (observation[6] - 8, observation[12]+6),
            (observation[7] - 8, observation[13]+6),
            (observation[8] - 8, observation[14]+6),
            (observation[9] - 8, observation[15]+6)
        ],
        'advantages': [
            (observation[10] - 8, observation[16]+6),
        ]
    }
    return parameter


def make_Influence_map(new_img, parameter):
    for i in parameter['enemies']:
        img = np.zeros((210, 160, 3), np.uint8)
        cv2.circle(img, i, 35, [0, 0, 16], -1)
        cv2.circle(img, i, 30, [0, 0, 32], -1)
        cv2.circle(img, i, 25, [0, 0, 64], -1)
        cv2.circle(img, i, 3, [0, 0, 255], -1)
        new_img = cv2.add(new_img, img)

    for i in parameter['advantages']:
        img = np.zeros((210, 160, 3), np.uint8)
        cv2.circle(img, i, 35, [16, 0, 0], -1)
        cv2.circle(img, i, 30, [32, 0, 0], -1)
        cv2.circle(img, i, 25, [64, 0, 0], -1)
        cv2.circle(img, i, 3, [255, 0, 0], -1)
        new_img = cv2.add(new_img, img)
    # BGR->RGB
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    return new_img


def get_state(new_img, observation):

    param = pacman_obs_Ram(observation)
    img = make_Influence_map(new_img, param)
    img = img[0:172, :]
    img = cv2.resize(img, dsize=(84, 84), interpolation=cv2.INTER_AREA)

    cv2.imshow("a", img)

    img = img.transpose(2, 0, 1)
    # img = img[:, 0:172]

    img = torch.from_numpy(img).type(torch.float32)

    # resize = T.Compose([T.ToPILImage(), T.Resize(40, T.InterpolationMode.BICUBIC), T.ToTensor()])
    # # resize = T.Compose([T.Resize(40, T.InterpolationMode.BICUBIC), T.ToTensor()])
    # img = resize(img)

    # save_image(img, 'image_name.png')

    return img.unsqueeze(0).to(device)