import random

import torch
import torch.nn as nn

from torchattacks.attack import Attack


# 2022 CVPR 随机方差减少集成对抗性攻击以提高对抗性

# MI-FGSM
class SVRE_MI_FGSM(Attack):
    def __init__(self, ens_model, eps=16/255, alpha=16/255/10, steps=10, decay=1.0, m=16, beta=16/255/10, u=1.0):
        super().__init__("SVR_MI_FGSM", ens_model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        # SVRE
        self.m = m
        self.beta = beta
        self.u = u

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # grad = np.zeros(shape=batch_shape, dtype=np.float32)
        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        # x = np.copy(images)
        adv_images = images.clone().detach()

        models = ["inc_v3", "inc_v4", "inc_res_v2", "res_152"]

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # Calculate the gradient of the ensemble model
            # noise_ensemble = sess.run(grad_ensemble, feed_dict={x_input: x, y_input: labels})
            ensOutputs = self.model(adv_images)
            ensCost = loss(ensOutputs, labels)
            ensGrad = torch.autograd.grad(ensCost, adv_images, retain_graph=False, create_graph=False)[0]

            # Stochastic variance reduction via M updates
            # x_before = np.copy(x)
            # 初始化
            # x_inner = np.copy(x)
            accuImages = adv_images.clone().detach()
            # grad_inner = np.zeros_like(x)
            accuGrad = torch.zeros_like(images).detach().to(self.device)

            # SVRG
            for _ in range(self.m):
                accuImages.requires_grad = True

                # 随机挑选一个单模型
                flag = random.choice(models)
                self.model.flag = flag

                # noise_x_inner = sess.run(grad_single, feed_dict={x_input: x_inner, y_input: labels})
                # 计算内循环梯度
                tempOuts = self.model(accuImages)
                tempCost = loss(tempOuts, labels)
                tempGrad = torch.autograd.grad(tempCost, accuImages, retain_graph=False, create_graph=False)[0]

                # noise_x = sess.run(grad_single, feed_dict={x_input: x, y_input: labels})
                # 单模型梯度
                adv_images.requires_grad = True
                singleOutputs = self.model(adv_images)
                singleCost = loss(singleOutputs, labels)
                singleGrad = torch.autograd.grad(singleCost, adv_images, retain_graph=False, create_graph=False)[0]

                # noise_inner = noise_x_inner - (noise_x - noise_ensemble)
                innerGrad = tempGrad - (singleGrad - ensGrad)
                innerGrad = innerGrad / torch.mean(torch.abs(innerGrad), dim=(1, 2, 3), keepdim=True)

                # Update the inner gradient by momentum
                # grad_inner = momentum * grad_inner + noise_inner
                accuGrad = self.u * accuGrad + innerGrad

                accuImages = accuImages.detach() + self.beta * accuGrad.sign()
                delta = torch.clamp(accuImages - images, min=-self.eps, max=self.eps)
                accuImages = torch.clamp(images + delta, min=0, max=1).detach()

            self.model.flag = 'ens'

            accuGrad = accuGrad / torch.mean(torch.abs(accuGrad), dim=(1, 2, 3), keepdim=True)

            grad = self.decay * momentum + accuGrad
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images




















