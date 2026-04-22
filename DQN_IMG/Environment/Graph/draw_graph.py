"""
성능 평가 함수 개발 및 공유(학습시 스테이지 클리어 여부, 자체 Score 변화,
 loss 변화, 매 step별 최대/최소/평균 계산 시간 및 step별 계산시간 그래프 출력필요). 그래프 출력 기능 제공 필요

done episode: 에피소드가 끝날 경우는 목숨 3개가 아니라 1개. -> 변경 필요
슈퍼 모드로 변경해서 고스트를 먹는 것보다 모든 dot과 enery pill을 모두 먹는것을 우선적으로 선택 -> 변경필요.
스테이지가 클리어 했을 때 어떤 값을 출력하는지 확인 필요.

1) 학습시 스테이지 클리어 여부
-> 그래프 (x, y): 경과한 에피소드, 완료 (Ture or False))

2) 자체 reward 변화 (x, y)
-> 에피소드 당 스코어 그래프 (episode, reward)

3) loss 변화 (x, y)
-> 에피소드 당 loss 그래프 (episode, loss)
"""
# ※Reward 그래프 먼저 그려야 함

import os

import matplotlib.pyplot as plt
import numpy as np

episode_rewards = []
avg_rewards = []
avg_episodes = []

losses = []
avg_losses = []

episode_steps = []
avg_steps = []


def average_reward(episode_count, episode_reward):
    episode_rewards.append(episode_reward)

    if episode_count % 20 == 0 or episode_count == 1:  # 20 에피소드마다 reward 평균 저장
        avg_episodes.append(episode_count)  # [1, 10, 20...]
        avg_reward = np.mean(episode_rewards)
        avg_rewards.append(avg_reward)
        episode_rewards.clear()

        if episode_count % 100 == 0:  # 100 에피소드마다 저장된 reward 평균 그래프 그리기
            reward_graph(avg_episodes, avg_rewards)


def reward_graph(avg_episodes, avg_rewards):
    x = avg_episodes
    y = avg_rewards

    plt.figure()
    plt.plot(x, y)
    plt.title('Reward Graph')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')

    path = os.path.dirname(os.path.realpath(__file__)) + '/reward_graph'
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.savefig(path + '/episode{}.png'.format(avg_episodes[-1]))

    plt.close()


def average_loss(episode_count, loss):
    losses.append(loss.item())

    if episode_count % 20 == 0 or episode_count == 1:
        avg_loss = np.mean(losses)
        avg_losses.append(avg_loss)
        losses.clear()

        if episode_count % 100 == 0:
            loss_graph(avg_episodes, avg_losses)


def loss_graph(avg_episodes, avg_losses):
    x = avg_episodes
    y = avg_losses

    plt.figure()
    plt.plot(x, y, color='red')
    plt.title('Loss Graph')
    plt.xlabel('Episode')
    plt.ylabel('Avg Loss')

    path = os.path.dirname(os.path.realpath(__file__)) + '/loss_graph'
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.savefig(path + '/episode{}.png'.format(avg_episodes[-1]))

    plt.close()


def step_avg(episode_count, episode_step):
    episode_steps.append(episode_step)

    if episode_count % 20 == 0 or episode_count == 1:
        avg_step = np.mean(episode_steps)
        avg_steps.append(avg_step)
        episode_steps.clear()

        if episode_count % 100 == 0:
            step_graph(avg_episodes, avg_steps)


def step_graph(avg_episodes, avg_steps):
    x = avg_episodes
    y = avg_steps

    plt.figure()
    plt.plot(x, y, color='orange')
    plt.title('Step Graph')
    plt.xlabel('Episode')
    plt.ylabel('Avg Step')

    path = os.path.dirname(os.path.realpath(__file__)) + '/step_graph'
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.savefig(path + '/episode{}.png'.format(avg_episodes[-1]))

    plt.close()


def stage_clear_graph(episode_count, done_stage):
    x = episode_count
    y = done_stage

    plt.figure()
    plt.plot(x, y)
    plt.title('Stage Clear Graph')
    plt.xlabel('episode_count')
    plt.ylabel('Done_stage')
    plt.yticks([0, 1], labels=['Fail', 'Clear'])

    path = os.path.dirname(os.path.realpath(__file__)) + '/clear_graph'
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.savefig(path + '/episode{}.png'.format(episode_count))

    plt.close()
