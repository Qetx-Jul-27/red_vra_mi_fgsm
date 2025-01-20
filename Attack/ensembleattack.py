import torchattacks
from torchattacks.attack import Attack
import torch
import torch.nn as nn
# import timm
from torchvision import models

from models.verify import get_model


class ensemble_model(nn.Module):
    def __init__(self, models_path, flag='ens'):
        # 调用父类的初始化（必须）
        super(ensemble_model, self).__init__()
        # 定义神经网络的相关操作
        # if models_path == '':
        #     self.inc_v3 = timm.create_model('inception_v3', pretrained=True)
        #     self.inc_v4 = timm.create_model('inception_v4', pretrained=True)
        #     self.inc_res_v2 = timm.create_model('inception_resnet_v2', pretrained=True)
        #     self.res_152 = models.resnet152(pretrained=True)
            # self.res_152 = timm.create_model('resnet152', pretrained=True)
        # else:
        self.inc_v3 = get_model('tf_inception_v3', models_path)
        self.inc_v4 = get_model('tf_inception_v4', models_path)
        self.inc_res_v2 = get_model('tf_inc_res_v2', models_path)
        self.res_152 = get_model('tf_resnet_v2_152', models_path)
        # 用来输出单个模型结果
        self.flag = flag

    # 前向传播：input -> forward -> output
    def forward(self, input):
        if self.flag == "inc_v3":
            inc_v3_output = self.inc_v3(input)
            return inc_v3_output
        elif self.flag == "inc_v4":
            inc_v4_output = self.inc_v4(input)
            return inc_v4_output
        elif self.flag == "inc_res_v2":
            inc_res_v2_output = self.inc_res_v2(input)
            return inc_res_v2_output
        elif self.flag == "res_152":
            res_152_output = self.res_152(input)
            return res_152_output
        else:
            inc_v3_output = self.inc_v3(input)
            inc_v4_output = self.inc_v4(input)
            inc_res_v2_output = self.inc_res_v2(input)
            res_152_output = self.res_152(input)
            ens_output = (inc_v3_output + inc_v4_output + inc_res_v2_output + res_152_output) / 4
            return ens_output

























