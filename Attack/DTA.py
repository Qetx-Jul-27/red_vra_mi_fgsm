import torch
import torch.nn as nn

from torchattacks.attack import Attack


# 2023 arvix 通过方向调整提高对抗性示例的可转移性

# NI-FGSM
class DTA_NI_FGSM(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0, k=10, u=0.8):
        super().__init__("DTA_NI_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        # DTA超参数
        # 内循环次数
        self.k = k
        # 内循环衰减因子
        self.u = u

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            # 内循环中的动量
            tempGrad = momentum.detach().to(self.device)
            # 记录内循环中的总梯度
            totalGrad = torch.zeros_like(images).detach().to(self.device)

            tempAdvImages = adv_images.clone().detach()

            for _ in range(self.k):
                # eq.12
                tempAdvImages.requires_grad = True
                tempAdvOutputs = self.model(tempAdvImages)
                tempAdvCost = loss(tempAdvOutputs, labels)
                tempAdvGrad = torch.autograd.grad(tempAdvCost, tempAdvImages, retain_graph=False, create_graph=False)[0]

                nesAdvImages = tempAdvImages.detach() + self.alpha * self.decay * tempAdvGrad

                # eq.11
                nesAdvImages.requires_grad = True
                nesAdvOutputs = self.model(nesAdvImages)
                nesAdvCost = loss(nesAdvOutputs, labels)
                nesAdvGrad = torch.autograd.grad(nesAdvCost, nesAdvImages, retain_graph=False, create_graph=False)[0]
                nesAdvGrad = nesAdvGrad / torch.mean(torch.abs(nesAdvGrad), dim=(1,2,3), keepdim=True)

                tempGrad = self.u * tempGrad + nesAdvGrad

                totalGrad = totalGrad + tempGrad

                # eq.13
                tempAdvImages = tempAdvImages.detach() + self.alpha / self.k * tempGrad.sign()
                delta = torch.clamp(tempAdvImages - images, min=-self.eps, max=self.eps)
                tempAdvImages = torch.clamp(images + delta, min=0, max=1).detach()

            # eq.10
            grad = self.decay * momentum + totalGrad / self.k
            momentum = grad

            # eq.4
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
