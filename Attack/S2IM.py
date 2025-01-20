import torch
from torch import nn
from torchattacks.attack import Attack
from util import DCT
import torch.nn.functional as F
import numpy as np
from scipy import stats as st


class S2_MI_FGSM(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0, N=20, rho=0.5):
        super().__init__("S2_MI_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        # S2MIM超参数
        self.N = N
        self.rho = rho

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            # adv_images.requires_grad = True

            # 频域变换
            totalGrad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                gauss = torch.randn(adv_images.size()[0], 3, 299, 299) * self.eps
                gauss = gauss.cuda()
                images_dct = DCT.dct_2d(adv_images + gauss).cuda()
                mask = (torch.rand_like(adv_images) * 2 * self.rho + 1 - self.rho).cuda()
                images_idct = DCT.idct_2d(images_dct * mask)

                images_idct.requires_grad = True
                outputs = self.model(images_idct)
                cost = loss(outputs, labels)
                tempGrad = torch.autograd.grad(cost, images_idct, retain_graph=False, create_graph=False)[0]

                totalGrad = totalGrad + tempGrad
            grad = totalGrad / self.N

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class S2_TI_MI_FGSM(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0, N=20, rho=0.5):
        super().__init__("S2_TI_MI_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        # S2MIM超参数
        self.N = N
        self.rho = rho
        # TIM
        self.kernel_name = 'gaussian'
        self.len_kernel = 15
        self.nsig = 3
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)

        # TIM
        stacked_kernel = self.stacked_kernel.to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            # 频域变换
            totalGrad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                gauss = torch.randn(adv_images.size()[0], 3, 299, 299) * self.eps
                gauss = gauss.cuda()
                images_dct = DCT.dct_2d(adv_images + gauss).cuda()
                mask = (torch.rand_like(adv_images) * 2 * self.rho + 1 - self.rho).cuda()
                images_idct = DCT.idct_2d(images_dct * mask)

                images_idct.requires_grad = True
                outputs = self.model(images_idct)
                cost = loss(outputs, labels)
                tempGrad = torch.autograd.grad(cost, images_idct, retain_graph=False, create_graph=False)[0]

                totalGrad = totalGrad + tempGrad
            grad = totalGrad / self.N

            # TIM
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel


class S2_TI_DI_MI_FGSM(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0, N=20, rho=0.5, resize_rate=0.9, diversity_prob=0.5):
        super().__init__("S2_TI_DI_MI_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        # DIM
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        # S2MIM超参数
        self.N = N
        self.rho = rho
        # TIM
        self.kernel_name = 'gaussian'
        self.len_kernel = 15
        self.nsig = 3
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)

        # TIM
        stacked_kernel = self.stacked_kernel.to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            # 频域变换
            totalGrad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                gauss = torch.randn(adv_images.size()[0], 3, 299, 299) * self.eps
                gauss = gauss.cuda()
                images_dct = DCT.dct_2d(adv_images + gauss).cuda()
                mask = (torch.rand_like(adv_images) * 2 * self.rho + 1 - self.rho).cuda()
                images_idct = DCT.idct_2d(images_dct * mask)

                images_idct.requires_grad = True
                outputs = self.model(self.input_diversity(images_idct))
                cost = loss(outputs, labels)
                tempGrad = torch.autograd.grad(cost, images_idct, retain_graph=False, create_graph=False)[0]

                totalGrad = totalGrad + tempGrad
            grad = totalGrad / self.N

            # TIM
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel


class S2_SI_MI_FGSM(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0, m=5, N=20, rho=0.5):
        super().__init__("S2_SI_MI_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m = m
        # S2MIM超参数
        self.N = N
        self.rho = rho

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            # 频域变换
            totalGrad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                gauss = torch.randn(adv_images.size()[0], 3, 299, 299) * self.eps
                gauss = gauss.cuda()
                images_dct = DCT.dct_2d(adv_images + gauss).cuda()
                mask = (torch.rand_like(adv_images) * 2 * self.rho + 1 - self.rho).cuda()
                images_idct = DCT.idct_2d(images_dct * mask)

                # SIM
                g = torch.zeros_like(images).detach().to(self.device)
                for i in torch.arange(self.m):
                    img = images_idct / torch.pow(2, i)

                    img.requires_grad = True
                    outputs = self.model(img)
                    cost = loss(outputs, labels)
                    g += torch.autograd.grad(cost, img, retain_graph=False, create_graph=False)[0]
                tempGrad = g / self.m

                totalGrad = totalGrad + tempGrad
            grad = totalGrad / self.N

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class S2_TI_SI_MI_FGSM(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0, m=5, N=20, rho=0.5):
        super().__init__("S2_TI_SI_MI_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        # TIM
        self.kernel_name = 'gaussian'
        self.len_kernel = 15
        self.nsig = 3
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        # SIM
        self.m = m
        # S2MIM超参数
        self.N = N
        self.rho = rho

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        # TIM
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            # 频域变换
            totalGrad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                gauss = torch.randn(adv_images.size()[0], 3, 299, 299) * self.eps
                gauss = gauss.cuda()
                images_dct = DCT.dct_2d(adv_images + gauss).cuda()
                mask = (torch.rand_like(adv_images) * 2 * self.rho + 1 - self.rho).cuda()
                images_idct = DCT.idct_2d(images_dct * mask)

                # SIM
                g = torch.zeros_like(images).detach().to(self.device)
                for i in torch.arange(self.m):
                    img = images_idct / torch.pow(2, i)

                    img.requires_grad = True
                    outputs = self.model(img)
                    cost = loss(outputs, labels)
                    g += torch.autograd.grad(cost, img, retain_graph=False, create_graph=False)[0]
                tempGrad = g / self.m

                totalGrad = totalGrad + tempGrad
            grad = totalGrad / self.N

            # TIM
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel


class S2_VT_NI_FGSM(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=5, decay=1.0, N=20, beta=3/2, N2=20, rho=0.5):
        super().__init__("S2_VT_NI_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.beta = beta
        self.resize_rate = 0.9
        # S2MIM超参数
        self.N2 = N2
        self.rho = rho

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        momentum = torch.zeros_like(images).detach().to(self.device)

        v = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            # adv_images.requires_grad = True
            nes_images = adv_images + self.decay * self.alpha * momentum

            # 频域变换
            totalGrad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N2):
                gauss = torch.randn(nes_images.size()[0], 3, 299, 299) * self.eps
                gauss = gauss.cuda()
                images_dct = DCT.dct_2d(nes_images + gauss).cuda()
                mask = (torch.rand_like(nes_images) * 2 * self.rho + 1 - self.rho).cuda()
                images_idct = DCT.idct_2d(images_dct * mask)

                images_idct.requires_grad = True
                outputs = self.model(images_idct)
                cost = loss(outputs, labels)
                tempGrad = torch.autograd.grad(cost, images_idct, retain_graph=False, create_graph=False)[0]

                totalGrad = totalGrad + tempGrad
            adv_grad = totalGrad / self.N2

            grad = (adv_grad + v) / torch.mean(torch.abs(adv_grad + v), dim=(1,2,3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + \
                                  torch.randn_like(images).uniform_(-self.eps*self.beta, self.eps*self.beta)
                neighbor_images.requires_grad = True
                outputs = self.model(neighbor_images)

                # Calculate loss
                cost = loss(outputs, labels)
                GV_grad += torch.autograd.grad(cost, neighbor_images,
                                               retain_graph=False, create_graph=False)[0]
            # obtaining the gradient variance
            v = GV_grad / self.N - adv_grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class S2_TI_SI_VT_NI_FGSM(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=0.0, m=5, N=20, beta=3/2, N2=20, rho=0.5):
        super().__init__("S2_TI_SI_VT_NI_FGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.kernel_name = 'gaussian'
        self.len_kernel = 15
        self.nsig =3
        self.m = m
        self.N = N
        self.beta = beta
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        # S2MIM超参数
        self.N2 = N2
        self.rho = rho

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)
        v = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            # adv_images.requires_grad = True
            nes_images = adv_images + self.decay * self.alpha * momentum

            # 频域变换
            totalGrad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N2):
                gauss = torch.randn(nes_images.size()[0], 3, 299, 299) * self.eps
                gauss = gauss.cuda()
                images_dct = DCT.dct_2d(nes_images + gauss).cuda()
                mask = (torch.rand_like(nes_images) * 2 * self.rho + 1 - self.rho).cuda()
                images_idct = DCT.idct_2d(images_dct * mask)

                # SIM
                g = torch.zeros_like(images).detach().to(self.device)
                for i in torch.arange(self.m):
                    img = images_idct / torch.pow(2, i)

                    img.requires_grad = True
                    outputs = self.model(img)
                    cost = loss(outputs, labels)
                    g += torch.autograd.grad(cost, img, retain_graph=False, create_graph=False)[0]
                tempGrad = g / self.m

                totalGrad = totalGrad + tempGrad
            grad = totalGrad / self.N2

            # TIM
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)

            grad = (grad + v) / torch.mean(torch.abs(grad + v), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(images).detach().to(self.device)
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + \
                                  torch.randn_like(images).uniform_(-self.eps * self.beta, self.eps * self.beta)
                neighbor_images.requires_grad = True
                outputs = self.model(neighbor_images)

                # Calculate loss
                cost = loss(outputs, labels)
                GV_grad += torch.autograd.grad(cost, neighbor_images,
                                               retain_graph=False, create_graph=False)[0]
            # obtaining the gradient variance
            v = GV_grad / self.N - grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen, kernlen)) * 1.0 / (kernlen * kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1 - np.abs(np.linspace((-kernlen + 1) / 2, (kernlen - 1) / 2, kernlen) / (kernlen + 1) * 2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel


































