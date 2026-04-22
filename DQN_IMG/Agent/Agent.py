# Model 폴더 안에 있는 sample_model.py를 import 합니다.
from Agent.Model.model_git import Network

# Environment 폴더 안에 있는 based_Environment.py를 import 합니다.
import Environment.Environment as Env

import Environment.Influence_map as Influ
import Environment.Graph.draw_graph as Graph


import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import deque

"""사용할 그래픽카드에 따라 cuda 번호 설정"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deque_data = deque(maxlen=10000)

main_model = None
target_model = None

parameter = {
    'game_name': 'Breakout-v0',
    'epsilon_start': 0.99,
    'epsilon_end': 0.05,
    'epsilon_decay': ['Manual', 100],
    'gamma': 0.99,
    'Update': 10,
    'experience': 100,
    'Use_Influencemap': False,
    'load_file': None,
    'render': True
}


def _build_model(input_data, output_data) -> Network:
    """
    딥러닝 모델을 생성합니다. 모델은 DQN, A3C, PPO 등이 될 수 있습니다.

    :param input_data:
    :return: Model
    """
    return Network(input_data, output_data).to(device)


def _save_data(*args) -> None:
    """
    DQN 학습을 위한 데이터를 저장합니다.

    :param args: (state, next_state, action, reward)
    :return: None
    """
    set = []
    for i in args:
        if i is None:
            set.append(i)
        else:
            set.append(i.to('cpu'))

    deque_data.append(set)


def _load_data(sampling_size):
    Data = random.sample(deque_data, sampling_size)
    Full_data = []
    for args in Data:
        batch_data = (
            args[0].to(device),
            args[1].to(device) if not args[1] is None else None,
            args[2].to(device),
            args[3].to(device),
        )
        Full_data.append(batch_data)
    return tuple(Full_data)


def _save_train_model(episodes) -> None:
    """
    모델을 저장합니다.

    :param episodes: 모델의 episodes 수에 따라 파일 이름을 변경합니다.
    :return: None
    """

    torch.save(main_model, 'model_{}__Influ({}).pt'.format(episodes, parameter['Use_Influencemap']))
    print("save_complete")


def load_model() -> Network:
    """
    저장한 모델을 불러와 main_model에 저장합니다.
    target_model를 업데이트하고 모든 웨이트와 바이어스를 복사합니다.

    :return: model
    """
    try:
        model = torch.load("{}.pt".format(parameter['load_file']), map_location=device)
        return model
    except:
        print("로드할 파일이 존재하지 않습니다.")


def _action_predict(state, step):
    """
   현재 state를 통해 action을 배출합니다.

   :param state: 현재 상태
   """
    epsilon_start, epsilon_end, epsilon_decay = parameter['epsilon_start'], parameter['epsilon_end'], parameter['epsilon_decay'][1]

    sample = random.random()
    eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * step / epsilon_decay)

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            return main_model(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(0, 9)]], device=device, dtype=torch.long)


def _optimize_model():
    """
    Model을 최적화 합니다.
    """
    if len(deque_data) <= parameter['experience']:
        return

    batch = _load_data(parameter['experience'])

    non_final_mask = torch.tensor([experience[1] is not None for experience in batch], dtype=torch.bool)
    non_final_next_states = torch.cat([experience[1] for experience in batch if experience[1] is not None]).type(torch.float32) / 255.0

    state_batch = torch.cat([experience[0] for experience in batch]).type(torch.float32) / 255.0
    action_batch = torch.cat([experience[2] for experience in batch])
    reward_batch = torch.cat([experience[3] for experience in batch])

    state_action_values = main_model(state_batch).gather(1, action_batch)

    with torch.no_grad():
        target_model.eval()
        next_state_values = torch.zeros(parameter['experience'], device=device)
        next_state_values[non_final_mask] = target_model(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * parameter['gamma']) + reward_batch

    criterion = nn.HuberLoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optim.Adam(main_model.parameters(), lr=0.0001).zero_grad()
    loss.backward()
    optim.Adam(main_model.parameters(), lr=0.0001).step()

    return loss


def train(max_steps=100000, train_num=100000) -> None:
    """
    model을 학습합니다.
    <실행 과정>
    1. 환경 생성 및 환경 초기화
    2. 환경 observation 저장 -> Env에서 observation 변환 -> 학습 -> 행동 -> [2] 반복

    :param max_steps: 최대 step 수. 무한하게 행동하는 것을 막습니다. (Default: 100,000)
    :param train_num: 학습 횟수를 지정할 수 있습니다. (Default: 1,000,000)
    :return: None
    """
    if parameter['epsilon_decay'][0] == 'Auto':
        parameter['epsilon_decay'][1] = int(train_num / 0.95)  # 트레이닝 횟수 증가에 따른 decay 변화 자동화

    env = Env.make_env(parameter['game_name'])
    steps = 0

    global main_model, target_model

    Use_Influ = parameter['Use_Influencemap']
    if Use_Influ:
        def get_state(observation):
            return Influ.get_state(base_img, observation)
        input_channel = 3

    else:
        def get_state(observation):
            return Env.get_state(observation)
        input_channel = 1

    target_model = _build_model(input_channel, 5).to(device)
    if parameter['load_file'] is not None:
        main_model = load_model()
        main_model.train()
    else:
        main_model = _build_model(input_channel, 5,).to(device)

    target_model.load_state_dict(main_model.state_dict())

    for i in tqdm(range(1, train_num + 1)):
        if Use_Influ:
            base_img = Influ.create_base_image()
        obs = env.reset()  # 게임 초기화

        ingame_steps = 0
        total_reward = 0
        state = get_state(obs)

        while True:
            action = _action_predict(state, steps)  # 행동 계산
            obs, reward, done, info = env.step(action)
            if parameter['render']:
                env.render()
            if Use_Influ:
                base_img = Influ.renewal_map(base_img, obs)

            ingame_steps += 1
            steps += 1  # 스텝 수 증가

            reward = torch.tensor([reward], device=device)

            next_state = get_state(obs)

            done = done or not info['ale.lives'] == 3 or ingame_steps >= max_steps
            if done:
                next_state = None
                reward = -10
                total_reward += reward
                Graph.average_reward(i, total_reward)

            torch_reward = torch.tensor([reward], device=device)
            _save_data(state, next_state, action, torch_reward)

            state = next_state

            _optimize_model()

            if done:
                break

        if i % 500 == 0:
            _save_train_model(i)

        if i % parameter['Update'] == 0:
            target_model.load_state_dict(main_model.state_dict())


def test():
    game = Env.make_env(parameter['game_name'])

    global main_model
    main_model = load_model()
    main_model.eval()

    Use_Influ = parameter['Use_Influencemap']
    if Use_Influ:
        def get_state(observation):
            return Influ.get_state(base_img, observation)
    else:
        def get_state(observation):
            return Env.get_state(observation)

    done = True
    total_reward = 0

    while True:
        if done == True:
            print(total_reward)
            if Use_Influ:
                base_img = Influ.create_base_image()
            total_reward = 0
            step = 0
            observation = game.reset()
            last = get_state(observation)

        state = get_state(observation)
        action = main_model(state).max(1)[1].view(1, 1)

        observation, reward, done, info = game.step(action)

        done = done or info['ale.lives'] < 3

        if reward > 50:
            reward = 0
        total_reward += reward

        step += 1
        if Use_Influ:
            base_img = Influ.renewal_map(base_img, observation)
        game.render()

