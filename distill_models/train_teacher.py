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
from networks.get_model import get_model
from torch.optim.lr_scheduler import StepLR
from src import utils


def train_epoch(epoch):
    start = time.time()
    T_model.train()
    for batch_index, (images, targets) in enumerate(train_loader):
        # Forward current model
        outputs = T_model(images.to(device))
        loss = criterion(outputs, targets.to(device))
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(T_model.parameters(), args.clipping)
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
    T_model.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    with torch.no_grad():
        for (images, targets) in eval_loader:
            outputs = T_model(images.to(device))
            loss = criterion(outputs, targets.to(device))
            test_loss += loss.item()
            _, preds = outputs.max(1)
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

    # 杂项参数 miscellaneous args
    parser.add_argument('--gpu', type=int, default=5,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=1993,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--save-models', action='store_false',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--t-path', type=str, default='/',
                        help='Save teacher model pth file(default=%(default)s)')

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
    parser.add_argument('--network', default='ResNet18', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    parser.add_argument('--get-features', action='store_true',
                        help='return features or not')

    # 训练参数 training args
    parser.add_argument('--nepochs', default=1, type=int, required=False,
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
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'

    train_loader, test_loader, valid_loader, num_classes = get_loaders(dataset=args.datasets,
                                                                       batch_size=args.batch_size,
                                                                       num_workers=args.num_workers,
                                                                       validation=args.validation)

    best_acc = 0.0
    T_model = get_model(args.network, num_classes, args.pretrained, get_features=args.get_features)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(T_model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=45, gamma=0.1)
    T_model.to(device)

    print("***************************START TRAINING!***************************")
    for epoch in range(1, args.nepochs + 1):
        # Train
        train_epoch(epoch)
        scheduler.step()

        # 如果使用验证集，每个epoch之后在验证集上计算loss和accuracy
        if args.validation > 0:
            acc = eval_training(epoch, valid_loader)

            # start to save best performance model after learning rate decay to 0.01
            if epoch > 45 and best_acc < acc:
                path = os.path.join(args.t_path, args.network + 'pth')
                print('saving weights file to {}'.format(path))
                torch.save(T_model.state_dict(), path)
                best_acc = acc
                continue

    print("***************************TEST!***************************")
    eval_training(args.nepochs, test_loader)

    if args.save_models:
        path = os.path.join(args.t_path, args.network + '.pth')
        print('saving weights file to {}'.format(path))
        torch.save(T_model.state_dict(), path)
