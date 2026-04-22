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
        # 기본 설정, train과 똑같이 맞출것
        'epsilon_start': 0.99,
        'epsilon_end': 0.05,
        'epsilon_decay': 100,
        'gamma': 0.99,
        'Update': 20,
        'experience': 10000,

        # 영향력 사용 유무, 게임 이름 변경, 모델 Load
        'game_name': 'MsPacman-ram-v4',
        'Use_Influencemap': True,
        'load_file': 'model_2000__Influ(True)'
    }

    Agent.test()
