import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """
    基础 SRCNN 模型：
    conv1 + ReLU + conv2 + ReLU + conv3
    """
    def __init__(self):
        super(SRCNN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class SRCNN_MC(nn.Module):
    """
    带 MC-Dropout 的 SRCNN 模型。

    设计逻辑：
    1. 训练阶段 model.train() 时，Dropout 自动开启；
    2. 验证阶段 model.eval() + set_mc_dropout(False) 时，Dropout 关闭；
    3. MC 推理阶段 model.eval() + set_mc_dropout(True) 时，Dropout 重新开启。

    结构：
    conv1 -> ReLU -> Dropout
    conv2 -> ReLU -> Dropout
    conv3

    注意：
    不在 conv3 输出后使用 Dropout2d。
    因为 conv3 输出只有 1 个通道，如果使用 Dropout2d，可能整张输出图被随机置零。
    """

    def __init__(self, dropout_p: float = 0.10):
        super(SRCNN_MC, self).__init__()

        self.dropout_p = dropout_p
        self.mc_dropout = True

        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)

        self.relu = nn.ReLU(inplace=True)

    def set_mc_dropout(self, enabled: bool):
        """
        控制 eval 阶段是否强制开启 Dropout。

        enabled=False:
            eval 阶段 Dropout 关闭，用于验证集确定性评估。

        enabled=True:
            eval 阶段 Dropout 开启，用于 MC-Dropout 推理。
        """
        self.mc_dropout = enabled

    def forward(self, x):
        use_dropout = self.training or self.mc_dropout

        x = self.relu(self.conv1(x))
        x = F.dropout(x, p=self.dropout_p, training=use_dropout)

        x = self.relu(self.conv2(x))
        x = F.dropout(x, p=self.dropout_p, training=use_dropout)

        x = self.conv3(x)
        return x