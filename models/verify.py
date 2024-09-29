"""Implementation of evaluate attack result."""
import os

# import timm
import torch
import torchvision.models
from torch.autograd import Variable as V
from torch import nn
# from torch.autograd.gradcheck import zero_gradients
from tqdm import tqdm

from models.Normalize import Normalize, TfNormalize
# from loader import ImageNet
from torch.utils.data import DataLoader
from models.torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# tf2torch
def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')
        return None

    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        net.KitModel(model_path),
    )

    return model


def verify(root, model_name, models_path, device=torch.device('cpu')):
    # model = get_model(model_name, models_path).eval().to(device)

    model = timm.create_model(model_name, pretrained=True).eval().to(device)

    # 数据集预处理
    batch_size = 1

    transform = transforms.Compose([
        transforms.Resize([299, 299]),
        transforms.ToTensor(),
    ])
    # 加载数据集
    dataset = datasets.ImageFolder(root=root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    count = 0
    total = 0

    for i, (images, labels) in enumerate(tqdm(dataloader)):
        if i >= 20:
            break
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            predict = outputs.max(1)[1]
            count = count + (predict == labels).detach().sum().item()
            total = total + batch_size

    print(model_name + '  acu = {:.2%}'.format(count / total))


def main():

    model_names = ['tf_inception_v3', 'tf_inception_v4', 'tf_inc_res_v2',
                   'tf_resnet_v2_50', 'tf_resnet_v2_101', 'tf_resnet_v2_152',
                   'tf_adv_inception_v3', 'tf_ens3_adv_inc_v3', 'tf_ens4_adv_inc_v3', 'tf_ens_adv_inc_res_v2']
    models_path = r''

    # model_names = ['inception_v3', 'inception_v4', 'inception_resnet_v2',
    #                'resnet50', 'resnet101', 'resnet152',
    #                'adv_inception_v3', 'ens_adv_inception_resnet_v2']

    root = r""

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    # for model_name in model_names:
    #     verify(root, model_name, models_path, device)
    #     print("===================================================")

    verify(root, 'ens_adv_inception_resnet_v2', models_path, device)


if __name__ == '__main__':
    main()