import os
import Agent.Agent as Agent

# ####
# # 특정 GPU를 사용하기 위해서는 아래의 주석을 해지
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPU 0
# # os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # GPU 1
# # os.environ["CUDA_VISIBLE_DEVICES"] = '2'  # GPU 2
# # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # GPU 1과 2를 병렬로 사용
# ####


if __name__ == '__main__':
    Agent.parameter = {
        'game_set': ['MsPacman-ram-v4', 10],  # [game_name, frame_skip]
        'Use_Influencemap': True,  # Influence map 사용 여부(사용시 atari pacman 환경은 ram으로 변경할 것)

        'epsilon_start': 0.99,
        'epsilon_end': 0.05,
        'epsilon_decay': ['Auto', None],
        'gamma': 0.99,
        'Update': 20,
        'experience': 32,  # 배치 사이즈(학습할 이미지 개수)

        'load_file': None,
        'render': True
    }

    Agent.train(max_steps=10000, train_num=50000)






