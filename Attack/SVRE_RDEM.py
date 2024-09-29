import random
from scipy import stats as st
import numpy as np
import torch
import torch.nn as nn
from torchattacks.attack import Attack
import torch.nn.functional as F
from torchvision import transforms


class SVRE_RDE_MI_FGSM(Attack):
    def __init__(self, model, eps=16/255, alpha=16/255/10, steps=10, decay=1.0, en=5, M=16, beta=16/255/10, u=1.0):
        super().__init__("SVRE_RDE_MI_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.en = en
        # SVRE
        self.M = M
        self.beta = beta
        self.u = u

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        trans = [transforms.RandomHorizontalFlip(p=0.7),
                 transforms.RandomRotation(degrees=(-20, 20)),
                 transforms.RandomAffine(0, translate=(0.2, 0.2)),
                 transforms.RandomAffine(0, scale=(0.8, 1)),
                 transforms.RandomAffine(0, shear=(-20, 20))]

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # Calculate the gradient of the ensemble model
            ens_outputs = torch.zeros((images.shape[0], 1000)).to(self.device)
            for i in range(self.en):
                # 从变换池中随机挑选多个变换
                tran = transforms.Compose(random.choices(trans, k=3))
                output = self.model(tran(adv_images))
                ens_outputs = ens_outputs + output

            ens_outputs = ens_outputs / self.en
            ens_cost = loss(ens_outputs, labels)
            ens_grad = torch.autograd.grad(ens_cost, adv_images, retain_graph=False, create_graph=False)[0]

            # Stochastic variance reduction via M updates
            accu_images = adv_images.clone().detach()
            accu_grad = torch.zeros_like(images).detach().to(self.device)

            # SVRG
            for _ in range(self.M):
                accu_images.requires_grad = True

                # 随机挑选一个单模型
                tran = transforms.Compose(random.choices(trans, k=3))

                # 计算内循环梯度
                temp_outs = self.model(tran(accu_images))
                temp_cost = loss(temp_outs, labels)
                temp_grad = torch.autograd.grad(temp_cost, accu_images, retain_graph=False, create_graph=False)[0]

                # 单模型梯度
                adv_images.requires_grad = True
                single_outputs = self.model(tran(adv_images))
                single_cost = loss(single_outputs, labels)
                single_grad = torch.autograd.grad(single_cost, adv_images, retain_graph=False, create_graph=False)[0]

                inner_grad = temp_grad - (single_grad - ens_grad)
                inner_grad = inner_grad / torch.mean(torch.abs(inner_grad), dim=(1, 2, 3), keepdim=True)

                grad_variance = torch.var(inner_grad, dim=(1, 2, 3), keepdim=True)
                # Update the inner gradient by momentum
                accu_grad = self.u * accu_grad + inner_grad

                accu_images = accu_images.detach() + self.beta * accu_grad.sign()
                delta = torch.clamp(accu_images - images, min=-self.eps, max=self.eps)
                accu_images = torch.clamp(images + delta, min=0, max=1).detach()

            accu_grad = accu_grad / torch.mean(torch.abs(accu_grad), dim=(1, 2, 3), keepdim=True)

            grad = accu_grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images,1


