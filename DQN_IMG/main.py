import os
import Environment.Environment  as Env

####
# 특정 GPU를 사용하기 위해서는 아래의 주석을 해지
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID' #GPU 번호를 직접 선택해서 사용하겠다는 의미
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPU 0
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # GPU 1
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'  # GPU 2
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # GPU 1과 2를 병렬로 사용
####


def run():
    Env.run(
        load_model_name=None,
        game_name='Breakout-v0',
    )


if __name__ == '__main__':
    run()
