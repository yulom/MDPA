from model.model_utils import SSA, LightweightConv1d, Mixer1D
from model.grl import WarmStartGradientReverseLayer
#network architecture modified from：SST-DPN https://github.com/hancan16/SST-DPN
import torch.nn as nn
import torch

class Efficient_Encoder(nn.Module):

    def __init__(
        self,
        samples,
        chans,
        F1=16,
        F2=36,
        time_kernel1=75,
        pool_kernels=[50, 100, 250],
    ):
        super().__init__()

        self.time_conv = LightweightConv1d(
            in_channels=chans,
            num_heads=1,
            depth_multiplier=F1,
            kernel_size=time_kernel1,
            stride=1,
            padding="same",
            bias=True,
            weight_softmax=False,
        )
        self.ssa = SSA(samples, chans * F1)

        self.chanConv=nn.Conv1d(
                chans * F1,
                F2,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        self.batchNorm1d = nn.BatchNorm1d(F2)
        self.dBatchNorm1d = []
        for i in range(9):
            self.dBatchNorm1d += [nn.BatchNorm1d(F2).cuda()]
        self.elu = nn.ELU()

        self.mixer = Mixer1D(dim=F2, kernel_sizes=pool_kernels)

    def forward(self, x,x_domain=None,dBatchNorm=False):

        x = self.time_conv(x)
        # print(x.shape)
        x, _ = self.ssa(x)
        # print(x.shape)
        x_chan = self.chanConv(x)
        if dBatchNorm:
            y = torch.zeros_like(x_chan)
            for i in x_domain.unique():
                x_ = x_chan[x_domain==i]
                # print(x_.shape)
                y[x_domain==i] = self.dBatchNorm1d[i-1](x_)
            x_chan = self.elu(y)
        else:
            x_chan = self.batchNorm1d(x_chan)
        # print(x_chan.shape)
        feature = self.mixer(x_chan)
        # print(feature.shape)

        # feature = self.linear(feature)
        return feature

# class GradientReverseLayer(torch.autograd.Function):
#     def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
#         self.iter_num = iter_num
#         self.alpha = alpha
#         self.low_value = low_value
#         self.high_value = high_value
#         self.max_iter = max_iter

#     def forward(self, input):
#         self.iter_num += 1
#         output = input * 1.0
#         return output

#     def backward(self, grad_output):
#         self.coeff = np.float(
#             2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
#                         self.high_value - self.low_value) + self.low_value)
#         return -self.coeff * grad_output

class EEGEncoder(nn.Module):

    def __init__(
        self,
        chans,
        samples,
        num_classes=4,
        F1=9,
        F2=48,
        time_kernel1=75,
        pool_kernels=[50, 100, 200],
    ):
        super().__init__()
        self.encoder = Efficient_Encoder(
            samples=samples,
            chans=chans,
            F1=F1,
            F2=F2,
            time_kernel1=time_kernel1,
            pool_kernels=pool_kernels,
        )
        self.features = None

        x = torch.ones((1, chans, samples))
        out = self.encoder(x)
        feat_dim = out.shape[-1]
        # self.grl_layer = GradientReverseLayer()
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        
        class ClassifyHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.proto = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
                nn.init.kaiming_normal_(self.proto)
            def forward(self,x,wog = False):
                if wog:
                    return -torch.cdist(x, self.proto.detach(), p=2)
                else:
                    return -torch.cdist(x, self.proto, p=2)
        self.classifyHead = ClassifyHead()
        self.AdvHead = ClassifyHead()

    def get_features(self):
        if self.features is not None:
            return self.features
        else:
            raise RuntimeError("No features available. Run forward() first.")

    def forward(self, x,x_domain):

        features = self.encoder(x,x_domain,dBatchNorm=True)
        self.features = features
        logits = self.classifyHead(features,wog=False)
        adv_features = self.grl_layer(features)
        adv_logits = self.AdvHead(adv_features,wog=False)

        return logits,features,adv_logits