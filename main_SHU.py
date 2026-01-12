import argparse
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True
import math
from model.model_utils import SSA, LightweightConv1d, Mixer1D
from model.grl import WarmStartGradientReverseLayer
from protoLoss import TDF_Loss,mdd_adv_loss
from tqdm import tqdm
import logging

#EDPNet
class Efficient_Encoder(nn.Module):

    def __init__(
        self,
        samples,
        chans,
        subn=9,
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
        for i in range(subn):
            self.dBatchNorm1d += [nn.BatchNorm1d(F2).cuda()]
        self.elu = nn.ELU()

        self.mixer = Mixer1D(dim=F2, kernel_sizes=pool_kernels)

        # self.linear = nn.Linear(1072, 128)

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


class EDPNet(nn.Module):

    def __init__(
        self,
        chans,
        samples,
        subn = 9,
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
            subn=subn,
            F1=F1,
            F2=F2,
            time_kernel1=time_kernel1,
            pool_kernels=pool_kernels,
        )
        self.features = None

        x = torch.ones((1, chans, samples))
        out = self.encoder(x)
        feat_dim = out.shape[-1]
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
                # return torch.einsum("bd,cd->bc", x, self.isp)
        self.classifyHead = ClassifyHead()
        self.AdvHead = ClassifyHead()

    def get_features(self):
        if self.features is not None:
            return self.features
        else:
            raise RuntimeError("No features available. Run forward() first.")

    def forward(self, x,x_domain,dBatchNorm=True):

        features = self.encoder(x,x_domain,dBatchNorm=True)
        self.features = features
        logits = self.classifyHead(features,wog=False)
        adv_features = self.grl_layer(features)
        adv_logits = self.AdvHead(adv_features,wog=False)

        return logits,features,adv_logits

class TLExP():
    def __init__(self, args):
        super(TLExP, self).__init__()
        self.args = args
        test_i = args.target_sub
        trainSetI = generate_trainSetI(test_i)
        if test_i < 10:
            test_i = '00' + str(test_i)
        else:
            test_i = '0'+str(test_i)
        testSetI = [test_i]
        self.batch_size = args.batch_size
        # self.batch_size = 72*2
        self.n_epochs = args.n_epochs
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        # self.clip_value = 0.01
        self.trainSetI = trainSetI
        self.testSetI = testSetI
        self.targetDomainLabel = int(self.testSetI[0][-2:])
        self.seti = [self.testSetI[0][-2:]]
        for i in self.trainSetI:
            self.seti.append(i[-2:])
        self.SetIs = set(self.seti)
        self.trainNum = len(self.SetIs)
        self.proto_detach = args.proto_detach
        self.method = args.method

        self.start_epoch = 0
        self.root = '/standard_WBCIC_data/'

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        print(self.trainNum)

        self.model = EDPNet(chans=58, samples=1000, num_classes=3,subn=self.trainNum).cuda()
        self.model = self.model.cuda()
        now = time.localtime()
        now_t = time.strftime("%Y-%m-%d-%H:%M",now)
        log_path = 'logs_SHU/'+testSetI[0]+args.method+'new2_w_tops_'+str(args.w_tdf)+'lr_proto_'+str(args.lr_proto)+'with_AdaBN'+str(args.with_AdaBN)+'proto_detach'+str(args.proto_detach)+'e_'+str(args.n_epochs)+'.log'
        logging.basicConfig(
            filename=log_path,
            filemode='a',
            format='%(asctime)s - %(message)s',
            level=logging.INFO
        )
        logging.info(self.args)

    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug_TL(self, timg, label):  
        aug_data = []
        aug_label = []

        for cls0 in torch.unique(label[:,0]):
            for cls1 in torch.unique(label[:,1]):
                conbinedCls = torch.tensor((cls0,cls1))
                cls_idx0 = np.where(label[:,0] == conbinedCls[0])
                cls_idx1 = np.where(label[:,1] == conbinedCls[1])
                cls_idx=np.intersect1d(cls_idx0,cls_idx1)
                #print(cls_idx)
                tmp_data = timg[cls_idx]
                tmp_label = label[cls_idx,:]
                tmp_aug_data = np.zeros((int(self.batch_size / (len(torch.unique(label[:,0]))*len(torch.unique(label[:,1])))),20, 1000))
                for ri in range(int(self.batch_size / (len(torch.unique(label[:,0]))*len(torch.unique(label[:,1]))))):
                    for rj in range(8):
                        rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                        tmp_aug_data[ri, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :,
                                                                      rj * 125:(rj + 1) * 125]
                aug_data.append(tmp_aug_data)
                aug_label.append(tmp_label[0,:].repeat(tmp_aug_data.shape[0],1))

        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label).cuda()
        aug_label = aug_label.long()
        aug_label[:,0] = aug_label[:,0]
        return aug_data, aug_label

    def get_source_data(self):
        print('load train data...')
        for i in range(len(self.trainSetI)):
            filename = self.trainSetI[i]
            total_data = scipy.io.loadmat(self.root+filename+'.mat')
            if i==0:
                train_data = total_data['data']
                train_label = total_data['label']
                train_seti = np.full_like(total_data['label'],int(filename[-2:]))
            else:
                train_data = np.concatenate((train_data,total_data['data']),axis=2)
                train_label = np.concatenate((train_label,total_data['label']))
                train_seti = np.concatenate((train_seti,np.full_like(total_data['label'],int(filename[-2:]))))
        print('load test data...')
        for i in range(len(self.testSetI)):
            filename = self.testSetI[i]
            total_data = scipy.io.loadmat(self.root+filename+'.mat')
            if i==0:
                test_data = total_data['data']
                test_label = total_data['label']
                test_seti = np.full_like(total_data['label'],int(filename[-2:]))
            else:
                test_data = np.concatenate((test_data,total_data['data']),axis=2)
                test_label = np.concatenate((test_label,total_data['label']))
                test_seti = np.concatenate((test_seti,np.full_like(total_data['label'],int(filename[-2:]))))

        test_data = np.transpose(test_data, (2, 1, 0))
        train_data = np.transpose(train_data, (2, 1, 0))

        # x_train = np.expand_dims(x_train, axis=1)
        # y_train = np.transpose(y_train)
        # seti_train = np.transpose(seti_train)
        # train_data = np.expand_dims(train_data, axis=1)
        train_label = np.transpose(train_label[:,0])
        train_seti = np.transpose(train_seti[:,0])
        # test_data = np.expand_dims(test_data, axis=1)
        test_label = np.transpose(test_label[:,0])
        test_seti = np.transpose(test_seti[:,0])
        print('data load finished')
        return train_data,train_label,train_seti,test_data,test_label,test_seti
        # return x_train,y_train,seti_train,test_data,test_label,test_seti


    def train(self):
        self.train_data, self.train_label, self.train_seti, self.test_data, self.test_label, self.test_seti = self.get_source_data()
        # return self.train_data, self.train_label, self.train_seti, self.test_data, self.test_label, self.test_seti
        
        train_data = torch.from_numpy(self.train_data)
        train_label = torch.from_numpy(self.train_label - 1)
        train_seti = torch.from_numpy(self.train_seti)
        self.train_data = train_data
        self.train_label = torch.stack((train_label,train_seti),dim=1)
        rand_index = torch.randperm(self.train_label.size(0))
        # val_ratio = 0.2
        val_ratio = 0.125
        val_index = int(self.train_label.size(0)*val_ratio)
        self.val_data = self.train_data[rand_index[:val_index]]
        self.val_label = self.train_label[rand_index[:val_index]]
        self.train_data = self.train_data[rand_index[val_index:]]
        self.train_label = self.train_label[rand_index[val_index:]]
        # img = img[label<=1,:,:]
        # seti = seti[label<=1]
        # label = label[label<=1]
        dataset = torch.utils.data.TensorDataset(self.train_data, self.train_label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        
        # return self.dataloader

        test_data = torch.from_numpy(self.test_data)
        test_label = torch.from_numpy(self.test_label - 1)
        test_seti = torch.from_numpy(self.test_seti)
        
        self.test_data = test_data
        self.test_label =torch.stack((test_label,test_seti),dim=1)
        # randi = torch.randperm(self.test_label.shape[0])
        # self.test_label_pred = self.test_label[randi,:]
        # test_data = test_data[test_label<=1,:,:]
        # test_seti = test_seti[test_label<=1]
        # test_label = test_label[test_label<=1]
        test_dataset = torch.utils.data.TensorDataset(self.test_data,self.test_label)
        if self.args.with_aug:
            self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size*2, shuffle=True)
        else:
            self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
    
        cls_proto_params = self.model.classifyHead.proto
        adv_proto_params = self.model.AdvHead.proto
        
        other_params = [param for name, param in self.model.named_parameters() if name != 'classifyHead.proto' and name != 'AdvHead.proto']
        self.optimizer_proto = torch.optim.Adam([cls_proto_params, adv_proto_params], lr=self.lr*10, betas=(self.b1, self.b2))

        self.optimizer_other = torch.optim.Adam(other_params, lr=self.lr, betas=(self.b1, self.b2))

 
        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(torch.stack((test_label,test_seti),dim=1).type(self.LongTensor))
        
        self.val_data = Variable(self.val_data.type(self.Tensor))
        self.val_label = Variable(self.val_label.type(self.LongTensor))

        bestAcc = 0
        bestAcc_val = 0
        finalAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0


        self.mcc_ratio = 0.
        for e in tqdm(range(self.n_epochs)):
            # in_epoch = time.time()
            self.model.train()
            # if e%10 == 0:
            #     s_img,s_label = next(iter(self.dataloader))
            #     s_img = Variable(s_img.cuda().type(self.Tensor))
            #     s_label = Variable(s_label.cuda().type(self.LongTensor))
            #     aug_img,aug_label = self.interaug_TL(self.train_data, self.train_label)
            #     # return img,label,aug_img,aug_label
            #     s_img = torch.cat((s_img,aug_img))
            #     s_label = torch.cat((s_label,aug_label))
            #     outputs,feature = self.model(s_img,s_label[:,1])
                
            #     centroids = torch.zeros(self.model.classifyHead.isp.shape).cuda()
            #     # 计算每一类的中心点
            #     for i in range(4):
            #         class_features = feature[s_label[:,0] == i].detach()
            #         centroids[i] = class_features.mean(dim=0)
            #     self.model.classifyHead.isp.data = centroids
            
            for ee in range(8):
            # for ee in range(4):
                # return self.dataloader,self.test_dataloader
                s_img,s_label = next(iter(self.dataloader))
                s_img = Variable(s_img.cuda().type(self.Tensor))
                s_label = Variable(s_label.cuda().type(self.LongTensor))
                if self.args.with_aug:
                    aug_img,aug_label = self.interaug_TL(self.train_data, self.train_label)
                    # return img,label,aug_img,aug_label
                    s_img = torch.cat((s_img,aug_img))
                    s_label = torch.cat((s_label,aug_label))

                t_img,t_label = next(iter(self.test_dataloader))
                t_img = Variable(t_img.cuda().type(self.Tensor))
                t_label = Variable(t_label.cuda().type(self.LongTensor))
                # return t_img
                minlen = min(s_img.shape[0],t_img.shape[0])
                # print(s_img.shape,t_img.shape,minlen)
                # print(s_label,t_label)
                s_img = s_img[:minlen,:,:]
                s_label = s_label[:minlen,:]
                t_img = t_img[:minlen,:,:]
                t_label = t_label[:minlen,:]
                img = torch.cat((s_img.cuda(),t_img.cuda()))
                label= torch.cat((s_label.cuda(),t_label.cuda()))
                # print(s_label.shape,t_label.shape)
                domains = label[:,1]
                # return img,label
                outputs,feature,adv_logits = self.model(img,domains)
                # return feature
                # return outputs,features,label
                y_s, y_t = outputs.chunk(2, dim=0)
                y_s_adv, y_t_adv = adv_logits.chunk(2, dim=0)

                features_s, features_t = feature.chunk(2, dim=0)
                classfyout = y_s
                trainlabel = s_label

                cls_loss = self.criterion_cls(classfyout, s_label[:,0]) 
                adv_loss = mdd_adv_loss(y_s,y_t,y_s_adv, y_t_adv)

                tdf_loss = TDF_Loss(y_t,self.model.classifyHead.proto,proto_detach=self.proto_detach)
                self.tdf_ratio = 1-np.exp(1*e/(e-self.n_epochs+1e-9))

                w_tdf = 0.01
                w_adv = 0.01
                loss = cls_loss + w_tdf*tdf_loss + w_adv*adv_loss

                self.optimizer_proto.zero_grad()
                self.optimizer_other.zero_grad()
                loss.backward()
                self.optimizer_proto.step()
                self.optimizer_other.step()
            
            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                with torch.no_grad():
                    Cls,_,_ = self.model(test_data, test_label[:,1])
    
                    loss_test = self.criterion_cls(Cls, test_label[:,0])
                    y_pred = torch.max(Cls, 1)[1]
                    acc = float((y_pred == test_label[:,0]).cpu().numpy().astype(int).sum()) / float(test_label[:,0].size(0))
                    train_pred = torch.max(classfyout, 1)[1]
                    train_acc = float((train_pred == trainlabel[:,0]).cpu().numpy().astype(int).sum()) / float(trainlabel[:,0].size(0))

                    Cls,_,_ = self.model(self.val_data, self.val_label[:,1])
    
                    loss_val = self.criterion_cls(Cls, self.val_label[:,0])
                    y_pred = torch.max(Cls, 1)[1]
                    acc_val = float((y_pred == self.val_label[:,0]).cpu().numpy().astype(int).sum()) / float(self.val_label[:,0].size(0))
                    
                    num = num + 1
                    averAcc = averAcc + acc
                    if acc > bestAcc:
                        bestAcc = acc
                    if acc_val > bestAcc_val:
                        finalAcc = acc
                        bestAcc_val = acc_val
                        logging.info(f"best! Epoch {e}: Train loss={loss.detach().cpu().numpy():.4f} TDF_loss={tdf_loss.detach().cpu().numpy():.4f} Cls_lss={cls_loss.detach().cpu().numpy():.4f} Train accuracy={train_acc:.4f} Val accuracy={acc_val:.4f} Test accuracy={acc:.4f}")
                    
        averAcc = averAcc / num
        print('The test accuracy is:', finalAcc)
        print('The last accuracy is:', acc)
        print('The best accuracy is:', bestAcc)
        logging.info(f"The test accuracy is:{finalAcc} The last accuracy is:{acc} The best accuracy is:{bestAcc}")

        return bestAcc, averAcc, Y_true, Y_pred
    
def generate_trainSetI(exp_id):
    trainSetI = []
    for i in range(1, 12):
        if i == exp_id:
            continue
        if i < 10:
            trainSetI.append('00' + str(i))
        else:
            trainSetI.append('0'+str(i))
    return trainSetI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_sub", type=int, default=1)
    parser.add_argument("--lr_proto", type=float, default=1.)
    parser.add_argument("--with_AdaBN", type=bool, default=True)
    parser.add_argument("--w_tdf", type=float, default=0.01)
    parser.add_argument("--with_aug", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=2000)
    parser.add_argument("--proto_detach", type=bool, default=True)
    parser.add_argument("--NEW2", type=bool, default=True)
    parser.add_argument("--method", type=str, default='tdf')
    
    args = parser.parse_args()
    
    exp = TLExP(args)
    bestAcc, averAcc, Y_true, Y_pred= exp.train()