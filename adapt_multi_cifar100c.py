'''
python adapt_multi_cifar100c.py --dset cifar100c --gpu_id 2 --output_src ckps/source/ --output ckps/adapt --batch_size 200 --worker 4
'''
import argparse
import os, sys
import os.path as osp
import time
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader, Dataset
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

from robustbench.data import load_imagenetc, load_cifar100c
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=36, crop_size=32, alexnet=False):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=36, crop_size=32, alexnet=False):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args): 
    dset_loaders = {}
    x_test, y_test = load_cifar100c(10000, 5, args.t_dset_path, True, [args.name_tar])
    x_test, y_test = x_test[ args.batch_idx*200 : (args.batch_idx+1)*200 ], y_test[ args.batch_idx*200 : (args.batch_idx+1)*200 ]
    print(x_test.size(), y_test.size())
    
    class TempSet(Dataset):
        def __init__(self, data, labels, transform=None, transform1=None, target_transform=None):
            super(TempSet, self).__init__()
            self.data = data
            self.labels = labels

            self.transform = transform
            self.target_transform = target_transform
            self.transform1 = transform1

        def __getitem__(self, index):
            img = self.data[index]
            target = self.labels[index]

            if self.transform is not None:
                img1 = self.transform(img)
            if self.transform1 is not None:
                img2 = self.transform1(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.transform1 is not None:
                return [img1,img2], target, index
            else:
                return img1, target, index

        def __len__(self):
            return len(self.data)
  
    
    dataset = TempSet(x_test, y_test, transform=image_train())
    dset_loaders["target"] = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker)  
    dset_loaders["target_"] = DataLoader(dataset, batch_size=args.batch_size*3, shuffle=False, num_workers=args.worker)
    dataset = TempSet(x_test, y_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dataset, batch_size=args.batch_size*3, shuffle=False, num_workers=args.worker)
    
    return dset_loaders

def train_target(args):
    dset_loaders = data_load(args)
    # set base network
    netF_list = [network.ResNextBase() for i in range(len(args.src))]  # 特征提取器

    w = 2*torch.rand((len(args.src),))-1  # 权重？
    print(w)

    netB_list = [network.feat_bottleneck(type=args.classifier, feature_dim=netF_list[i].in_features, bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    netC_list = [network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    netG_list = [network.scalar(w[i]).cuda() for i in range(len(args.src))]
    print(netG_list)

    param_group = []
    for i in range(len(args.src)):  # 加载源模型
        modelpath = args.output_dir_src[i] + '/source_F.pt'
        print(modelpath)
        netF_list[i].load_state_dict(torch.load(modelpath))  # 三个源模型的特征提取器的集合
        netF_list[i].eval()
        for k, v in netF_list[i].named_parameters():
            param_group += [{'params':v, 'lr':args.lr * args.lr_decay1}]

        modelpath = args.output_dir_src[i] + '/source_B.pt'
        print(modelpath)
        netB_list[i].load_state_dict(torch.load(modelpath))
        netB_list[i].eval()
        for k, v in netB_list[i].named_parameters():
            param_group += [{'params':v, 'lr':args.lr * args.lr_decay2}]

        modelpath = args.output_dir_src[i] + '/source_C.pt'
        print(modelpath)
        netC_list[i].load_state_dict(torch.load(modelpath))  # 分类器不更新
        netC_list[i].eval()
        for k, v in netC_list[i].named_parameters():
            v.requires_grad = False

        for k, v in netG_list[i].named_parameters():  # 权重也要更新
            param_group += [{'params':v, 'lr':args.lr}]
    
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    c = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            initc = []
            all_feas = []
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
                temp1, temp2 = obtain_label(dset_loaders['target_'], netF_list[i], netB_list[i], netC_list[i], args)
                temp1 = torch.from_numpy(temp1).cuda()
                temp2 = torch.from_numpy(temp2).cuda()
                initc.append(temp1)
                all_feas.append(temp2)
                netF_list[i].train()
                netB_list[i].train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        outputs_all = torch.zeros(len(args.src), inputs_test.shape[0], args.class_num)
        weights_all = torch.ones(inputs_test.shape[0], len(args.src))
        outputs_all_w = torch.zeros(inputs_test.shape[0], args.class_num)
        init_ent = torch.zeros(1,len(args.src))

        for i in range(len(args.src)):
            features_test = netB_list[i](netF_list[i](inputs_test))
            outputs_test = netC_list[i](features_test)
            softmax_ = nn.Softmax(dim=1)(outputs_test)
            ent_loss = torch.mean(loss.Entropy(softmax_))
            init_ent[:,i] = ent_loss
            weights_test = netG_list[i](features_test)
            outputs_all[i] = outputs_test
            weights_all[:, i] = weights_test.squeeze()

        z = torch.sum(weights_all, dim=1)
        z = z + 1e-16

        weights_all = torch.transpose(torch.transpose(weights_all,0,1)/z,0,1)
        outputs_all = torch.transpose(outputs_all, 0, 1)

        z_ = torch.sum(weights_all, dim=0)
        
        z_2 = torch.sum(weights_all)
        z_ = z_/z_2
    
        for i in range(inputs_test.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i],0,1), weights_all[i])
        
        if args.cls_par > 0:
            initc_ = torch.zeros(initc[0].size()).cuda()
            temp = all_feas[0]
            all_feas_ = torch.zeros(temp[tar_idx, :].size()).cuda()
            for i in range(len(args.src)):
                initc_ = initc_ + z_[i] * initc[i].float()
                src_fea = all_feas[i]
                all_feas_ = all_feas_ + z_[i] * src_fea[tar_idx, :]
            dd = torch.cdist(all_feas_.float(), initc_.float(), p=2)
            pred_label = dd.argmin(dim=1)
            pred_label = pred_label.int()
            pred = pred_label.long()
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_all_w, pred.cpu())
        else:
            classifier_loss = torch.tensor(0.0)

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_all_w)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:  #最后推理，报告结果
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
            acc, _ = cal_acc_multi(dset_loaders['test'], netF_list, netB_list, netC_list, netG_list, args)
            log_str = 'Iter:{}/{}; Accuracy = {:.2f}%'.format(iter_num, max_iter, acc)
            print(log_str+'\n')
            for i in range(len(args.src)):
                torch.save(netF_list[i].state_dict(), osp.join(args.output_dir, "target_F_" + str(i) + "_" + args.savename + ".pt"))
                torch.save(netB_list[i].state_dict(), osp.join(args.output_dir, "target_B_" + str(i) + "_" + args.savename + ".pt"))
                torch.save(netC_list[i].state_dict(), osp.join(args.output_dir, "target_C_" + str(i) + "_" + args.savename + ".pt"))
                torch.save(netG_list[i].state_dict(), osp.join(args.output_dir, "target_G_" + str(i) + "_" + args.savename + ".pt"))
            if iter_num == max_iter:
                return acc

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs.float()))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])

    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str+'\n')
    #return pred_label.astype('int')
    return initc,all_fea

def cal_acc_multi(loader, netF_list, netB_list, netC_list, netG_list, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs_all = torch.zeros(len(args.src), inputs.shape[0], args.class_num)
            weights_all = torch.ones(inputs.shape[0], len(args.src))
            outputs_all_w = torch.zeros(inputs.shape[0], args.class_num)
            
            for i in range(len(args.src)):
                features = netB_list[i](netF_list[i](inputs))
                outputs = netC_list[i](features)
                weights = netG_list[i](features)
                outputs_all[i] = outputs
                weights_all[:, i] = weights.squeeze()

            z = torch.sum(weights_all, dim=1)
            z = z + 1e-16

            weights_all = torch.transpose(torch.transpose(weights_all,0,1)/z,0,1)
            print(weights_all.mean(dim=0))
            outputs_all = torch.transpose(outputs_all, 0, 1)

            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i],0,1), weights_all[i])

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--t', type=int, default=0, help="target") ## Choose which domain to set as target {0 to len(names)-1}
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=32, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-caltech', choices=['office', 'office-home', 'office-caltech', 'imagenetc', 'cifar100c'])
    parser.add_argument('--lr', type=float, default=1*1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--batch_idx', type=int, default=-1, help="")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='ckps/adapt_ours')
    parser.add_argument('--output_src', type=str, default='ckps/source')
    args = parser.parse_args()
    
    args.dset = 'cifar100c'
    names = 'gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression'.split(', ')
    print(names)
    args.class_num = 100

    args.src = ['gaussian_noise', 'glass_blur', 'snow', 'jpeg_compression']

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.t_dset_path = '/home/yxue/datasets'

    args.output_dir_src = []  # 源模型的位置
    for i in range(len(args.src)):
        args.output_dir_src.append(osp.join(args.output_src, args.dset, args.src[i]))
    print(args.output_dir_src)

    args.savename = 'par_' + str(args.cls_par)

    for t in range(1):
        args.t = t
        args.name_tar = names[args.t]
        args.output_dir = osp.join(args.output, args.dset, names[args.t])

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)

        for i in range(10000//200):
            t1 = time.time()
            args.batch_idx = i
            acc = train_target(args)
            f = open(f'cifar100c-episodic_ggsj_target-{args.name_tar}.txt', 'a')
            f.write(f'{str(acc)}\n')
            f.close()
            t2 = time.time()
            print(f'batch time: {t2-t1}')

