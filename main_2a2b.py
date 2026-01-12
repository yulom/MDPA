import os
import logging
import argparse
import sys
import numpy as np
import torch
import scipy
from tqdm import tqdm
from torch.autograd import Variable
from model.sstdpn import EEGEncoder
from protoLoss import TDF_Loss,mdd_adv_loss

def setup_logging(args):
    log_dir = "log_new"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = f"{args.datasetname}_sub{args.subid}_tdf{args.w_tdf}_adv{args.w_adv}_epoch{args.n_epochs}_proto_detach{args.proto_detach}_data_aug{args.data_aug}_w_protonorm{args.w_protonorm}_tdf_t{args.tdf_t}_srcweight{args.srcweight}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
        ]
    )
    
    logger = logging.getLogger()
    
    logger.info("=" * 50)
    logger.info("config setting:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 50)
    
    return logger

class TLExP():
    def __init__(self, args):
        super(TLExP, self).__init__()
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.lr_proto = args.lr_proto
        self.b1 = args.b1
        self.b2 = args.b2
        self.trainSetI = args.trainSetI
        self.testSetI = args.testSetI
        self.w_tdf = args.w_tdf
        self.w_adv = args.w_adv
        self.tdf_t = args.tdf_t
        self.srcweight = args.srcweight
        self.w_protonorm=args.w_protonorm
        self.proto_detach = args.proto_detach
        self.targetDomainLabel = eval(args.testSetI[0][0])
        seti = [args.testSetI[0][0],]
        for i in args.trainSetI:
            seti.append(i[0])
        self.SetIs = set(seti)
        self.trainNum = len(self.SetIs)

        self.start_epoch = 0
        self.root = args.datasetdir
        self.data_aug = args.data_aug

        trainIs = ""
        for i in args.trainSetI:
            trainIs = trainIs+i

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.datasetname = args.datasetname
        if self.datasetname == '2a':
            self.num_classes=4
            self.model = EEGEncoder(chans=22, samples=1000, num_classes=4).cuda()
        elif self.datasetname == '2b':
            self.num_classes=2
            self.model = EEGEncoder(chans=3, samples=1000, num_classes=2).cuda()

        self.model = self.model.cuda()
        self.logger = setup_logging(args)

        

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
                tmp_aug_data = np.zeros((int(self.batch_size / (len(torch.unique(label[:,0]))*len(torch.unique(label[:,1])))),tmp_data.shape[1], tmp_data.shape[2]))
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
     
    def get_source_data(self,datasetname):
        filename = self.trainSetI[0]
        print('load train data...')
        for i in range(len(self.trainSetI)):
            filename = self.trainSetI[i]
            if datasetname == '2a':
                total_data = scipy.io.loadmat(self.root + 'rawA0'+filename+'.mat')
            elif datasetname == '2b':
                total_data = scipy.io.loadmat(self.root + 'B0'+filename+'.mat')
            if i==0:
                train_data = total_data['data']
                train_label = total_data['label']
                train_seti = np.full_like(total_data['label'],eval(filename[0]))
            else:
                train_data = np.concatenate((train_data,total_data['data']),axis=2)
                train_label = np.concatenate((train_label,total_data['label']))
                train_seti = np.concatenate((train_seti,np.full_like(total_data['label'],eval(filename[0]))))
        print('load test data...')
        for i in range(len(self.testSetI)):
            filename = self.testSetI[i]
            if datasetname == '2a':
                total_data = scipy.io.loadmat(self.root + 'rawA0'+filename+'.mat')
            elif datasetname == '2b':
                total_data = scipy.io.loadmat(self.root + 'B0'+filename+'.mat')
            if i==0:
                test_data = total_data['data']
                test_label = total_data['label']
                test_seti = np.full_like(total_data['label'],eval(filename[0]))
            else:
                test_data = np.concatenate((test_data,total_data['data']),axis=2)
                test_label = np.concatenate((test_label,total_data['label']))
                test_seti = np.concatenate((test_seti,np.full_like(total_data['label'],eval(filename[0]))))

        test_data = np.transpose(test_data, (2, 1, 0))
        train_data = np.transpose(train_data, (2, 1, 0))
        train_label = np.transpose(train_label[:,0])
        train_seti = np.transpose(train_seti[:,0])
        test_label = np.transpose(test_label[:,0])
        test_seti = np.transpose(test_seti[:,0])
        print('data load finished')
        return train_data,train_label,train_seti,test_data,test_label,test_seti


    def train(self):
        self.train_data, self.train_label, self.train_seti, self.test_data, self.test_label, self.test_seti = self.get_source_data(self.datasetname)
        
        train_data = torch.from_numpy(self.train_data)
        train_label = torch.from_numpy(self.train_label - 1)
        train_seti = torch.from_numpy(self.train_seti)
        self.train_data = train_data
        self.train_label = torch.stack((train_label,train_seti),dim=1)
        rand_index = torch.randperm(self.train_label.size(0))
        val_ratio = 0.125
        val_index = int(self.train_label.size(0)*val_ratio)
        self.val_data = self.train_data[rand_index[:val_index]]
        self.val_label = self.train_label[rand_index[:val_index]]
        self.train_data = self.train_data[rand_index[val_index:]]
        self.train_label = self.train_label[rand_index[val_index:]]
        dataset = torch.utils.data.TensorDataset(self.train_data, self.train_label)
        if self.data_aug:
            self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size*2, shuffle=True)
        else:
            self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(self.test_data)
        test_label = torch.from_numpy(self.test_label - 1)
        test_seti = torch.from_numpy(self.test_seti)
        
        self.test_data = test_data
        self.test_label =torch.stack((test_label,test_seti),dim=1)
        test_dataset = torch.utils.data.TensorDataset(self.test_data,self.test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        cls_proto_params = self.model.classifyHead.proto
        adv_proto_params = self.model.AdvHead.proto
        
        other_params = [param for name, param in self.model.named_parameters() if name != 'classifyHead.proto' and name != 'AdvHead.proto']
        self.optimizer_proto = torch.optim.Adam([cls_proto_params, adv_proto_params], lr=self.lr_proto, betas=(self.b1, self.b2))

        self.optimizer_other = torch.optim.Adam(other_params, lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(torch.stack((test_label,test_seti),dim=1).type(self.LongTensor))
        
        self.val_data = Variable(self.val_data.type(self.Tensor))
        self.val_label = Variable(self.val_label.type(self.LongTensor))

        bestAcc = 0
        bestAcc_val = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0
        bestLoss = 1e9


        self.mcc_ratio = 0.
        progress_bar = tqdm(range(self.n_epochs), desc="Training")
        ckpt_filename = f"{args.datasetname}_sub{args.subid}_tdf{args.w_tdf}_adv{args.w_adv}_epoch{args.n_epochs}_proto_detach{args.proto_detach}_data_aug{args.data_aug}_w_protonorm{args.w_protonorm}_tdf_t{args.tdf_t}_srcweight{args.srcweight}"
        for e in progress_bar:
            self.model.train()
            
            for ee in range(8):
                s_img,s_label = next(iter(self.dataloader))
                s_img = Variable(s_img.cuda().type(self.Tensor))
                s_label = Variable(s_label.cuda().type(self.LongTensor))
                if self.data_aug:
                    aug_img,aug_label = self.interaug_TL(self.train_data, self.train_label)
                    s_img = torch.cat((s_img,aug_img))
                    s_label = torch.cat((s_label,aug_label))
                t_img,t_label = next(iter(self.test_dataloader))
                t_img = Variable(t_img.cuda().type(self.Tensor))
                t_label = Variable(t_label.cuda().type(self.LongTensor))
                minlen = min(s_img.shape[0],t_img.shape[0])
                s_img = s_img[:minlen,:,:]
                s_label = s_label[:minlen,:]
                t_img = t_img[:minlen,:,:]
                t_label = t_label[:minlen,:]
                img = torch.cat((s_img.cuda(),t_img.cuda()))
                label= torch.cat((s_label.cuda(),t_label.cuda()))
                domains = label[:,1]
                outputs,feature,adv_logits = self.model(img,domains)
            
                y_s, y_t = outputs.chunk(2, dim=0)
                y_s_adv, y_t_adv = adv_logits.chunk(2, dim=0)

                
                features_s, features_t = feature.chunk(2, dim=0)
                classfyout = y_s
                trainlabel = s_label

                cls_loss = self.criterion_cls(classfyout, s_label[:,0]) 
                adv_loss = mdd_adv_loss(y_s,y_t,y_s_adv, y_t_adv,srcweight=self.srcweight)

                tdf_loss = TDF_Loss(y_t,self.model.classifyHead.proto,proto_detach=self.proto_detach,t=self.tdf_t)
                self.tdf_ratio = 1-np.exp(1*e/(e-self.n_epochs+1e-9))

                w_tdf = self.w_tdf
                w_adv = self.w_adv
                norms = torch.norm(self.model.classifyHead.proto, dim=1)
                normloss = norms.mean()
                loss = cls_loss + w_tdf*tdf_loss + w_adv*adv_loss + self.w_protonorm*normloss

                self.optimizer_proto.zero_grad()
                self.optimizer_other.zero_grad()
                loss.backward()
                self.optimizer_proto.step()
                self.optimizer_other.step()

            if (e + 1) % 1 == 0:
                self.model.eval()
                with torch.no_grad():
                    Cls,_,_ = self.model(test_data, test_label[:,1])
                    y_pred = torch.max(Cls, 1)[1]
                    acc = float((y_pred == test_label[:,0]).cpu().numpy().astype(int).sum()) / float(test_label[:,0].size(0))

                    Cls,_,val_adv = self.model(self.val_data, self.val_label[:,1])
    
                    cls_loss = self.criterion_cls(Cls, self.val_label[:,0])
                    y_pred = torch.max(Cls, 1)[1]
                    acc_val = float((y_pred == self.val_label[:,0]).cpu().numpy().astype(int).sum()) / float(self.val_label[:,0].size(0))
                    tdf_loss = TDF_Loss(Cls,self.model.classifyHead.proto,proto_detach=self.proto_detach)
                    adv_loss = mdd_adv_loss(y_s,Cls,y_s_adv, val_adv)
                    val_loss = cls_loss + w_tdf*tdf_loss + w_adv*adv_loss
                    
                    self.logger.info('Epoch: %d, Train loss: %.6f, TDF_loss: %.6f, cls_loss: %.6f, adv_loss: %.6f, norm_loss: %.6f, Val_loss: %.6f, Test accuracy: %.6f' % (
                        e,
                        loss.detach().cpu().numpy(),
                        tdf_loss.detach().cpu().numpy(),
                        cls_loss.detach().cpu().numpy(),
                        adv_loss.detach().cpu().numpy(),
                        normloss.detach().cpu().numpy(),
                        val_loss.detach().cpu().numpy(),
                        acc
                    ))
    
                    num = num + 1
                    averAcc = averAcc + acc
                    if acc > bestAcc:
                        bestAcc = acc
                        torch.save(self.model.state_dict(), 'ckpt_new/'+ckpt_filename+'best.pth')
                    if val_loss < bestLoss:
                        bestLoss = val_loss
                        bestAcc_val = acc
                        torch.save(self.model.state_dict(), 'ckpt_new/'+ckpt_filename+'val.pth')
                        self.logger.info('val best!')
            progress_bar.set_postfix({
                'Loss': f"{loss.detach().cpu().numpy():.4f}",
                'Acc': f"{acc:.4f}",
                'Best Val Acc': f"{bestAcc_val:.4f}",
                'Best Acc': f"{bestAcc:.4f}"
            })
        averAcc = averAcc / num
        self.logger.info('The last accuracy is:%.6f', acc)
        self.logger.info('The best accuracy is:%.6f', bestAcc)
        self.logger.info('The Val accuracy is:%.6f', bestAcc_val)
        torch.save(self.model.state_dict(), 'ckpt_new/'+ckpt_filename+'last.pth')

        return bestAcc, bestAcc_val, Y_true, Y_pred

def generate_datasets(number):
    all_numbers = [str(i) for i in range(1, 10)]
    train_set = [f"{num}T" for num in all_numbers if num != str(number)]
    test_set = [f"{number}E"]
    return train_set, test_set

def main(args):
    subid = args.subid
    trainSetI, testSetI = generate_datasets(subid)
    args.trainSetI = trainSetI
    args.testSetI = testSetI
    exp = TLExP(args)
    bestAcc, bestAcc_val, Y_true, Y_pred= exp.train()
    print('The Val accuracy is:', bestAcc_val)
    print(str(subid)+' finished')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetname', type=str, default='2a')
    parser.add_argument('--datasetdir', type=str, default='/standard_2a_data/')
    parser.add_argument('--subid', type=int, default=1)
    parser.add_argument('--w_tdf', type=float, default=0.01)
    parser.add_argument('--w_adv', type=float, default=0.01)
    parser.add_argument('--tdf_t', type=float, default=2.5)
    parser.add_argument('--srcweight', type=float, default=3.)
    parser.add_argument('--w_protonorm', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lr_proto', type=float, default=0.002)
    parser.add_argument('--b1', type=float, default=0.5)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--proto_detach', action='store_true', default=True)
    parser.add_argument('--data_aug', action='store_true', default=False)
    
    args = parser.parse_args()
    main(args)