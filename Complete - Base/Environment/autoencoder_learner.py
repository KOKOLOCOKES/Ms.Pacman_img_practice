import Environment.Environment as Env

import os
import torch
from torch import nn, optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from tqdm import tqdm
import glob
import random

epoch = 50
batch_size = 64
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

if USE_CUDA:
    print(' Use GPU')
else:
    print('Use CPU')


# def load_images(dir):
#     data = []
#     img_dir = glob.glob(dir + "/*.png")
#     random_img = random.sample(img_dir, len(img_dir))
#
#     for i in range(len(random_img)):
#         img = plt.imread(random_img[i])
#         img = img[:172, :, :]  # 모델에 따른 이미지 크기 자르기 (HxWxC)
#         data.append(img)
#     data = np.array(data, dtype=np.float32)
#
#     train_data = data[:3000]
#     test_data = data[3000:3500]
#
#     train_data = torch.tensor(train_data).float()
#     test_data = torch.tensor(test_data).float()
#
#     return train_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
        )

    def forward(self, x):
        encode_data = self.encoder(x)
        decode_data = self.decoder(encode_data)
        return encode_data, decode_data

def train(autoencoder, train_dataset):
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    autoencoder.train()
    for train_input in tqdm(train_dataset):
        # train_input = train_input.reshape(-1, 3, 160, 172).to(DEVICE)
        # train_input = np.transpose(train_input, (0, 3, 1, 2)).to(DEVICE)
        train_input = np.transpose(train_input, (0, 3, 1, 2)).to(DEVICE)
        encode_data, decode_data = autoencoder(train_input)
        loss = criterion(decode_data, train_input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(autoencoder, test_dataset):
    autoencoder.eval()
    test_loss = 0
    real_image = []
    gen_image = []

    criterion = nn.MSELoss()
    with torch.no_grad():
        for test_input in test_dataset:
            # f = test_input[:,:,:,0:1].squeeze().detach().cpu().numpy()
            # f = test_input.detach().cpu().numpy()
            test_input = np.transpose(test_input, (0, 3, 1, 2)).to(DEVICE)
            encode_data, decode_data = autoencoder(test_input)
            test_loss += criterion(decode_data, test_input).item()
            real_image.append(test_input.to('cpu'))
            gen_image.append(decode_data.to('cpu'))
    test_loss /= len(test_dataset.dataset)
    return test_loss, real_image, gen_image


def run(game_name, min_game, max_game):
    autoencoder = AutoEncoder().to(DEVICE)

    train_loader, test_loader = Env.get_dataset(game_name, min_game, max_game)
    train_dataset = data.DataLoader(dataset=train_loader, batch_size=batch_size)
    test_dataset = data.DataLoader(dataset=test_loader, batch_size=batch_size)

    for epo in range(1, epoch + 1):
        print("\n[epoch {}]".format(epo))
        train(autoencoder, train_dataset)

    test_loss, real_image, gen_image = evaluate(autoencoder, test_dataset)
    print("Test Loss: {:.4f} \n".format(test_loss))

    encoder = autoencoder.encoder
    decoder = autoencoder.decoder

    torch.save(encoder.state_dict(), './encoder_weight.pt')

    f, a = plt.subplots(2, 10, squeeze=False, figsize=(18, 2))
    Tensor_image = transforms.ToPILImage()
    for i in range(10):
        img = Tensor_image(real_image[0][i])
        a[0][i].imshow(img)
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

    for i in range(10):
        img = Tensor_image(gen_image[1][i])
        a[1][i].imshow(img)
        a[1][i].set_xticks(())
        a[1][i].set_yticks(())

    plt.show()
