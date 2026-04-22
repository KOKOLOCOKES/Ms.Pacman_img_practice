import os
import cv2
import gym
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = os.getcwd()


def make_env(game_set):
    """
    게임 환경을 구축합니다.

    <실행 과정>
    환경 구축 및 리턴

    :param game_name: 게임환경을 만들기 위해 필요한 이름입니다.
    :return: None
    """
    env = gym.make(game_set[0], frameskip=game_set[1])
    return env


def get_state(observation) -> torch.Tensor:  # 학습하는 데이터에 따라 제거 가능
    """
    observation을 model이 학습할 수 있는 state로 변경하는 함수입니다.

    :param obs: game observation
    :return: state는 tensor로 return합니다.
    """
    #
    # cv2.imshow("ss", observation)
    # cv2.waitKey(0)
    screen = observation[0:172, :]
    screen = cv2.resize(screen, dsize=(84, 84), interpolation=cv2.INTER_AREA)
    screen = screen.transpose((2, 0, 1))  # 3개
    # screen = screen[:, 0:173]

    # screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen).type(torch.float32)
    screen = T.Grayscale()(screen)
    #
    # resize = T.Compose([T.ToPILImage(), T.Grayscale(), T.Resize(40, T.InterpolationMode.BICUBIC), T.ToTensor()])
    # screen = resize(screen)

    return screen.unsqueeze(0).to(device)