import torch.nn as nn


class Network(nn.Module):
    def __init__(self, input_channel, output, size_w, size_h):
        """
            Sample 네트워크 입니다.
            Agent에 따라 Model은 달리질 수 있으나, Model은 언제나 Agent 안에서 호출되어야 합니다.

            :parameter input: 입력 값 (Default: 3 channel)
            :parameter output: 최종적으로 출력할 행동의 개수
            :parameter size_w: 이미지 가로 길이 (Default: None)
            :parameter size_h: 이미지 세로 길이 (Default: None)
        """

        super(Network, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(True)
            # nn.Conv2d(input_channel, 32, kernel_size=(5, 5)),
            # nn.ReLU(True),
            # nn.Conv2d(32, 64, kernel_size=(5, 5)),
            # nn.ReLU(True),
            # nn.Conv2d(64, 64, kernel_size=(5, 5)),
            # nn.ReLU(True)
        )

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(size_w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(size_h)))

        self.classifier = nn.Sequential(
            nn.Linear(convw * convh * 64, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
        )

        self.head = nn.Linear(32, output)

    def forward(self, x):
        x = self.cnn(x)
        x = nn.Flatten(1, -1)(x)
        x = self.classifier(x)

        x = self.head(x)

        return x
