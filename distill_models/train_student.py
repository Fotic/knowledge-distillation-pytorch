import os
import time
import torch
import argparse
import importlib
import numpy as np
from functools import reduce

from datasets.dataset import get_loaders
from datasets.dataset_config import dataset_config
from networks import tvmodels, allmodels
from methods import allmethods
from networks.get_model import get_model
from methods import *
from torch.optim.lr_scheduler import StepLR
import utils
import os

def train_epoch(epoch):
    start = time.time()
    S_model.train()
    for batch_index, (images, targets) in enumerate(train_loader):
        # Forward current model
        s_features, s_outputs = S_model(images.to(device))
        with torch.no_grad():
            t_features, t_outputs = T_model(images.to(device))

        kd_loss = kd_criterion(s_outputs, t_outputs)  # 这一项需要根据不同的蒸馏损失选择不同的输入
        if args.supervised:
            ce_loss = criterion(s_outputs, targets.to(device))
            loss = args.lamb * kd_loss + ce_loss
        else:
            loss = args.lamb * kd_loss
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(S_model.parameters(), args.clipping)
        optimizer.step()

        # 打印每次迭代的信息，包括当前epoch，已经训练的样本数量， loss 以及当前学习率
        # print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        #     loss.item(),
        #     optimizer.param_groups[0]['lr'],
        #     epoch=epoch,
        #     trained_samples=batch_index * args.batch_size + len(images),
        #     total_samples=len(cifar100_train_loader.dataset)
        # ))
    end = time.time()
    # 打印每个epoch消耗的时间
    # print('epoch {} training time consumed: {:.2f}s'.format(epoch, end - start))


def eval_training(epoch, eval_loader):
    start = time.time()
    S_model.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    with torch.no_grad():
        for (images, targets) in eval_loader:
            s_features, s_outputs = S_model(images.to(device))
            t_features, t_outputs = T_model(images.to(device))
            kd_loss = kd_criterion(s_outputs, t_outputs)  # 这一项需要根据不同的蒸馏损失选择不同的输入
            if args.supervised:
                ce_loss = criterion(s_outputs, targets.to(device))
                loss = args.lamb * kd_loss + ce_loss
            else:
                loss = args.lamb * kd_loss
            test_loss += loss.item()
            _, preds = s_outputs.max(1)
            correct += preds.eq(targets.to(device)).sum()

    end = time.time()
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.2f}%, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(eval_loader.dataset),
        (correct.float() / len(eval_loader.dataset)) * 100,
        end - start
    ))
    return correct.float() / len(eval_loader.dataset)


if __name__ == '__main__':
    # 参数解释说明Arguments
    parser = argparse.ArgumentParser(description='Distillation Test')
    # print(os.path.join(os.path.dirname(__file__)+ "\output_model"))

    # 杂项参数 miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=1993,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--save-models', action='store_false',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--s-path', type=str, default=os.path.join(os.path.dirname(__file__) + "\output_model"),
                        help='Save student model pth file (default=%(default)s)')
    parser.add_argument('--t-path', type=str,
                        default=os.path.join(os.path.dirname(__file__), "ResNet18.pt"),
                        help='Load teacher model pth file (default=%(default)s)')
    # 数据集参数 dataset args
    parser.add_argument('--datasets', default='cifar100', type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=256, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--validation', default=0.1, type=float, required=False,
                        help='validation 所占比例 (default=%(default)s)')

    # 网络模型参数 model args
    parser.add_argument('--s-network', default='mobilenet_v2', type=str, choices=allmodels,
                        help='学生网络(default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--t-network', default='ResNet18', type=str, choices=allmodels,
                        help='教师网络(default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    parser.add_argument('--get-features', action='store_false',
                        help='是否返回features')
    parser.add_argument('--supervised', action='store_false',
                        help='是否使用监督损失')

    # 选择使用的蒸馏方法 distillation method
    parser.add_argument('--KD-method', default='KD', type=str, choices=allmethods,
                        help='选择使用的蒸馏方法')
    parser.add_argument('--T', default=4, type=float, required=False,
                        help='KD用的温度系数 (default=%(default)s)')
    parser.add_argument('--dist', default=25, type=float, required=False,
                        help='RKD用的距离系数 (default=%(default)s)')
    parser.add_argument('--angle', default=50, type=float, required=False,
                        help='RKD用的角度系数 (default=%(default)s)')
    parser.add_argument('--lamb', default=1, type=float, required=False,
                        help='trade-off, 平衡交叉熵损失和蒸馏损失(default=%(default)s)')

    # 训练参数 training args
    parser.add_argument('--nepochs', default=5, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0002, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')

    args, extra_args = parser.parse_known_args()
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, fix_bn=args.fix_bn)

    utils.seed_everything(seed=args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda:0'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'

    train_loader, test_loader, valid_loader, num_classes = get_loaders(dataset=args.datasets,
                                                                       batch_size=args.batch_size,
                                                                       num_workers=args.num_workers,
                                                                       validation=args.validation)

    best_acc = 0.0
    S_model = get_model(args.s_network, num_classes, args.pretrained, args.get_features)
    T_model = get_model(args.t_network, num_classes, args.pretrained, args.get_features)
    T_model.load_state_dict(torch.load(args.t_path, map_location='cuda:0'), strict=True)
    T_model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(S_model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=45, gamma=0.1)
    S_model.to(device)
    T_model.to(device)

    kd_criterion = None
    if args.KD_method == "KD":
        kd_criterion = KD(args.T)
    elif args.KD_method == "RKD":
        kd_criterion = RKD(args.dist, args.angle)
    elif args.KD_method == "L2":
        kd_criterion = L2()
    elif args.KD_method == "PKT":
        kd_criterion = PKT()
    elif args.KD_method == "FitNets":
        kd_criterion = FitNet()

    print("***************************START TRAINING!***************************")
    for epoch in range(1, args.nepochs + 1):
        # Train
        train_epoch(epoch)
        scheduler.step()

        if args.validation > 0:
            acc = eval_training(epoch, valid_loader)
            # print(acc)
            # start to save best performance model after learning rate decay to 0.01
            if epoch > 45 and best_acc < acc:
                path = os.path.join(args.s_path, args.s_network + '.pth')
                print('saving weights file to {}'.format(path))
                torch.save(T_model.state_dict(), path)
                best_acc = acc
                continue

    if args.save_models:
        path = os.path.join(args.s_path, args.s_network + '.pth')
        print('saving weights file to {}'.format(path))
        torch.save(S_model.state_dict(), path)

        S_model.load_state_dict(torch.load(path))
        S_model.eval()
        cpu_tensor = torch.randn(32, 3, 32, 32)
        gpu_tensor = cpu_tensor.to(device)
        traced_script_module = torch.jit.trace(S_model, gpu_tensor)

        path = os.path.join(args.s_path, args.s_network + '.pt')
        print('saving pt file to {}'.format(path))
        traced_script_module.save( path )
