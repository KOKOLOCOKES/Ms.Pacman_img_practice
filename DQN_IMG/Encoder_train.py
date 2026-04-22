import os
import Environment.autoencoder_learner as AE

# ####
# # 특정 GPU를 사용하기 위해서는 아래의 주석을 해지
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPU 0
# # os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # GPU 1
# # os.environ["CUDA_VISIBLE_DEVICES"] = '2'  # GPU 2
# # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # GPU 1과 2를 병렬로 사용
# ####


if __name__ == '__main__':
    AE.run(
        game_name='MsPacman-v0',
        min_game=100,
        max_game=500
    )






