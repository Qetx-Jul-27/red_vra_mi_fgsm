import random
from scipy import stats as st
import numpy as np
import torch
import torch.nn as nn
from torchattacks.attack import Attack
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
class RDE_MI_FGSM(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=0.0, en=5, random_start=False):
        super().__init__("RDE_MI_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.en = en
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self._targeted:
        #     target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        # trans = transforms.RandomChoice([transforms.Grayscale(num_output_channels=3), transforms.RandomRotation(degrees=30),
        #                                  transforms.RandomHorizontalFlip(p=1), transforms.ColorJitter(brightness=(2, 2)),
        #                                  transforms.ColorJitter(contrast=(2, 2)), transforms.ColorJitter(saturation=(2, 2)),
        #                                  transforms.GaussianBlur(21, 1), transforms.RandomAffine(0, (0.1, 0.1)),
        #                                  transforms.RandomAffine(0, None, (0.7, 0.9)), transforms.RandomAffine(0, None, None, (10, 20))])

        # trans = [transforms.RandomHorizontalFlip(p=0.7), # 0.3 0.6959/0.7526
        #          transforms.Compose([transforms.RandomCrop(size=299 - 2 * 50), transforms.Pad(padding=50)]), # 0.8247/0.8814
        #          transforms.RandomRotation(degrees=(-40, 40)),  # 0.8814/0.9175
        #          transforms.RandomAffine(0, translate=(0.45, 0.45)),  # 0.9175/0.9381
        #          transforms.RandomAffine(0, scale=(0.7, 1)),  # 0.8247/0.8454
        #          transforms.RandomAffine(0, shear=(-30, 30))]  # 0.8299/0.8711

        # 0.9227/0.9381 0.8698/0.9010 0.5820/0.6296 0.2684/0.3211
        # 0.9021/0.9485 0.8698/0.9219 0.5873/0.6508 0.2947/0.3211

        trans = [transforms.RandomHorizontalFlip(p=0.7),
                 transforms.RandomRotation(degrees=(-20, 20)),
                 transforms.RandomAffine(0, translate=(0.2, 0.2)),
                 transforms.RandomAffine(0, scale=(0.8, 1)),
                 transforms.RandomAffine(0, shear=(-20, 20))]


        for _ in range(self.steps):
            adv_images.requires_grad = True

            # logit集成
            outputs = torch.zeros((images.shape[0], 1000)).to(self.device)
            for i in range(self.en):
                # 随机变换
                # tran = random.choice(trans)
                # 从变换池中随机挑选多个变换
                tran = transforms.Compose(random.choices(trans, k=3))

                output = self.model(tran(adv_images))
                outputs += output
            outputs /= self.en
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            # # prediction集成
            # log_softmax = nn.LogSoftmax(dim=1)
            # nll_loss = nn.NLLLoss()
            # prediction = torch.zeros((1, 1000)).to(self.device)
            # for i in range(self.en):
            #     tran = random.choice(trans)
            #     # tran = trans[i]
            #     output = self.model(tran(adv_images))
            #     pre = log_softmax(output)
            #     prediction += pre
            # prediction /= self.en
            # cost = nll_loss(prediction, labels)
            # grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            # # loss集成
            # all_loss = torch.tensor(0, dtype=torch.float).to(self.device)
            # for i in range(self.en):
            #     tran = random.choice(trans)
            #     output = self.model(tran(adv_images))
            #     cost = loss(output, labels)
            #     all_loss += cost
            # all_loss /= self.en
            # grad = torch.autograd.grad(all_loss, adv_images, retain_graph=False, create_graph=False)[0]

            # # grad集成
            # grads = torch.zeros_like(images).detach().to(self.device)
            # for i in range(self.en):
            #     # 从trans中随机选取一个操作
            #     tran = random.choice(trans)
            #     output = self.model(tran(adv_images))
            #     cost = loss(output, labels)
            #     grads += torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            # grad = grads / self.en


            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)

            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

