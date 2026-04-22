
import Agent

import os
import cv2
import gym
import time
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = os.getcwd()


def make_env(game_name):
    """
    게임 환경을 구축합니다.

    <실행 과정>
    환경 구축 및 리턴

    :param game_name: 게임환경을 만들기 위해 필요한 이름입니다.
    :return: None
    """
    env = gym.make(game_name, frameskip=10)
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


def plot_graph(current_episode, rewards, now='train') -> None:
    """
    게임 학습 상황을 그래프로 표현하는 함수입니다.

    ! 수정 필요, 사용 불가

    :param current_episode: 현재 진행중인 에피소드 번호
    :param rewards: Total reward
    :param now: train data, test data 구분
    :return: None
    """

    x = [i for i in range(current_episode)]
    x = x[current_episode - 100 if current_episode - 100 > 0 else 0: current_episode]
    rewards = rewards[len(rewards) - 100 if len(rewards) - 100 > 0 else 0: len(rewards)]

    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(x, rewards)

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤

    if current_episode % 50 == 0:
        path = PATH.join('/{}_data'.format(now))
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(path.join('/data_{}.png'.format(current_episode)))