import os
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import timm
import torch
import torchattacks
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from Attack import S2IM, customattack, SVRE, SVRE_RDEM
from Attack.ensembleattack import ensemble_model
from Defend.defendmodel import defend_model
from models.verify import get_model
from attack_methods import old_Attack, Attack


# 看看配置
print("CUDA is_available:", torch.cuda.is_available())
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")



model_names = ['tf_inception_v3', 'tf_inception_v4', 'tf_inc_res_v2',
               'tf_resnet_v2_50', 'tf_resnet_v2_101', 'tf_resnet_v2_152',
               'tf_adv_inception_v3', 'tf_ens3_adv_inc_v3', 'tf_ens4_adv_inc_v3', 'tf_ens_adv_inc_res_v2']
models_path = r'E:\WFZSS\pycharm项目\Model\npys'

inc_v3 = get_model('tf_inception_v3', models_path).eval().to(device)
# inc_v4 = get_model('tf_inception_v4', models_path).eval().to(device)
# inc_res_v2 = get_model('tf_inc_res_v2', models_path).eval().to(device)
res_50 = get_model('tf_resnet_v2_50', models_path).eval().to(device)
# res_101 = get_model('tf_resnet_v2_101', models_path).eval().to(device)
# res_152 = get_model('tf_resnet_v2_152', models_path).eval().to(device)
adv_inc_v3 = get_model('tf_adv_inception_v3', models_path).eval().to(device)
# ens3_adv_inc_v3 = get_model('tf_ens3_adv_inc_v3', models_path).eval().to(device)
# ens4_adv_inc_v3 = get_model('tf_ens4_adv_inc_v3', models_path).eval().to(device)
ens_adv_inc_res_v2 = get_model('tf_ens_adv_inc_res_v2', models_path).eval().to(device)
# ens_model = ensemble_model(models_path).eval().to(device)


# substitute_model
substitute_model = inc_v3

# victim_model 
# victim_model = [res_50]
# victim_model = [inc_v3, inc_v4, inc_res_v2, res_152, res_50, res_101]
# victim_model = [adv_inc_v3, ens3_adv_inc_v3, ens4_adv_inc_v3, ens_adv_inc_res_v2]
# victim_model = [inc_v3, inc_v4, inc_res_v2, res_152, res_50, res_101, adv_inc_v3, ens3_adv_inc_v3, ens4_adv_inc_v3, ens_adv_inc_res_v2]
victim_model = [ens_adv_inc_res_v2]


# 数据集预处理
transform = transforms.Compose([
    transforms.Resize([299, 299]),
    transforms.ToTensor(),
    # 标准化后无法展示对抗图像
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
# E:\WFZSS\pycharm项目
dataset_root = r'E:\WFZSS\pycharm项目\ImageNet\val_data_1000'
dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=5, shuffle=False)


# 实验次数 攻击图片数目
epoch = 1
number = 1000


# 防御方法 R&P HGD NRP JPEG DiffJPEG FD PD BDR TVM RS
# 防御模型：
# defend_path = r'E:\WFZSS\pycharm项目\Model\checkpoints'
# defend = defend_model(path=defend_path, defend=True, method="R&P").eval().to(device)
defend = None


atk = customattack.RDE_MI_FGSM(substitute_model, eps=16/255, alpha=16/255/10, steps=10, decay=1.0, en=5)

#atk = SVRE_RDEM.SVRE_RDE_MI_FGSM(substitute_model, eps=16/255, alpha=16/255/10, steps=10, decay=1.0, en=5, M=20, beta=16/255/10, u=1.0)


# begin_attack
start = time.time()
advs = Attack(atk, dataloader, defend, victim_model, epoch=epoch, number=number, device=device)
end = time.time()

print(f"--------------------攻击结束--------------------")

# 看看结果
print(f"{atk.model_name}使用{atk.attack}生成对抗样本")

length = len(victim_model)
acc = [0] * length
min = [1] * length
max = [0] * length

for i in range(epoch):
    for j in range(length):
        acc[j] = advs[i][j][0] / advs[i][j][1]
        if acc[j] < min[j]:
            min[j] = acc[j]
        if acc[j] > max[j]:
            max[j] = acc[j]

    print(f"第{i+1}轮攻击成功率：", end='')
    for k in range(length):
        print(f"%.4f" % (acc[k]), end=' ')
    print('')

print(f"共耗时%.2fmin" % ((end - start) / 60))

print(f"总共攻击了", end=' ')
for k in range(length):
    print(f"{advs[0][k][1]}", end=' ')
print(f"张图片")

print(f"最小攻击成功率/最大攻击成功率：", end='')
for k in range(length):
    print(f"%.4f/%.4f" % (min[k], max[k]), end=' ')