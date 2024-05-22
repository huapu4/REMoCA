# -*- coding: utf-8 -*-
"""
@Time    : 2021/10/28
@Author  : Lin Zhenzhe, Zhang Shuyi
"""
import glob
import os
import socket
import timeit
from datetime import datetime

import torch
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.dataset import VideoDataset
from network import C3D_model, P3D_model
from utils import compare_model
from config import Defalut_parameters

# Use GPU if available else revert to CPU
print('Please select a task for training:')
print(
    '1.fixation task (FT)\n2.saccadic task (ST)\n3.anti-saccade task (AST)\n4.horizontal smooth pursuit task (HSPT)\n5.vertical smooth pursuit task (VSPT)')
run_task = str(input())

Dp = Defalut_parameters()
dir_dict = Dp.get_task_dir(run_task)
torch.cuda.set_device(Dp.set_device)

device_ids = Dp.devices_ids
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device_ids)

nEpochs = Dp.nEpochs  # Number of epochs for training
resume_epoch = Dp.resume_epoch  # Default is 0, change if want to resume
useTest = Dp.useTest  # See evolution of the test set when training
nTestInterval = Dp.nTestInterval  # Run on test set every nTestInterval epochs
snapshot = Dp.snapshot  # Store a model every snapshot epochs
lr = Dp.lr  # Learning rate
prepare = Dp.prepare['train']
clip_len = Dp.clip_len
batch_size = Dp.batch_size

dataset = 'eyemovement'
num_classes = 2

save_dir = dir_dict['model_savepath']
data_dir = dir_dict['data_loadpath']
check_log_dir = dir_dict['check_log']

modelName = 'P3D'  # Options: C3D or P3D
saveName = modelName + '-' + run_task[1:]


def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    # model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
    model = P3D_model.P3D131(num_classes=num_classes)
    # train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
    #                 {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(model.parameters(),lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
        # model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # model = model.to(device)
    model = nn.DataParallel(model, device_ids).cuda()

    # criterion = criterion.to(device)
    criterion = criterion.cuda()

    log_dir = os.path.join(dir_dict['check_log'], datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(log_dir=log_dir)
    txt_file = os.path.join(log_dir, 'record.txt')
    txt_file2 = os.path.join(log_dir, 'record_wrong.txt')

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=data_dir, split='train', clip_len=clip_len, preprocess=prepare),
                                  batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(VideoDataset(dataset=data_dir, split='val', clip_len=clip_len, preprocess=prepare),
                                batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(VideoDataset(dataset=data_dir, split='test', clip_len=clip_len, preprocess=prepare),
                                 batch_size, shuffle=True, num_workers=0)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)
    cp_value = 0.15  # judge whether save model

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                # scheduler.step()
                model.train()
            else:
                model.eval()
            y_true, y_pred = [], []  # 0318
            for inputs, labels, filename in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                # inputs = Variable(inputs, requires_grad=True).to(device)
                # labels = Variable(labels).to(device)
                inputs = Variable(inputs, requires_grad=True).cuda()
                labels = Variable(labels).cuda()
                # filename = [item[0] for item in filename]
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.long())
                y_true.extend(labels.data.cpu().numpy())
                y_pred.extend(preds.data.cpu().numpy())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                # print(preds.tolist())
                # print(labels.data)
                try:
                    running_corrects += torch.sum(preds == labels.data)
                except:
                    running_corrects += torch.sum(preds.to(torch.int32) == labels)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()  # 0318
            tpr = tp / (tp + fn)  # 0318
            tnr = tn / (tn + fp)  # 0318
            print('tpr:{:.4f}'.format(tpr))  # 0318
            print('tnr:{:.4f}'.format(tnr))  # 0318
            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
                writer.add_scalar('data/train_tpr', tpr, epoch)
                writer.add_scalar('data/train_tnr', tnr, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)
                writer.add_scalar('data/val_tpr', tpr, epoch)
                writer.add_scalar('data/val_tnr', tnr, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
            if phase == 'train':
                with open(txt_file, 'a') as f:
                    f.writelines('Epoch_{}\n'.format(epoch))
                    f.writelines('train_acc:{:.4f}\n'.format(epoch_acc))
                    f.writelines('train_tpr:{:.4f}\n'.format(tpr))
                    f.writelines('train_tnr:{:.4f}\n'.format(tnr))
            else:
                with open(txt_file, 'a') as f:
                    f.writelines('val_acc:{:.4f}\n'.format(epoch_acc))
                    f.writelines('val_tpr:{:.4f}\n'.format(tpr))
                    f.writelines('val_tnr:{:.4f}\n'.format(tnr))
            f.close()
            if phase == 'val':
                # with open(txt_file2, 'a') as w:
                #     w.writelines('Epoch_{}_val\n'.format(epoch))
                #     for num in range(0, len(y_true)):
                #         if y_true[num] != y_pred[num]:
                #             w.writelines('{}\n'.format(filename[num]))
                # w.close()
                judge_save, cp_value = compare_model(tpr, tnr, cp_value)

        if judge_save:
            # if epoch % save_epoch == 0 and epoch != 0:
            # torch.save(model.state_dict(),os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth'))
            # torch.save({
            #     'epoch': epoch + 1,
            #     'state_dict': model.state_dict(),
            #     'opt_dict': optimizer.state_dict(),
            # }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth'))
            torch.save(model, os.path.join(save_dir, saveName + '_epoch-' + str(epoch) + '.pth'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, saveName + '_epoch-' + str(epoch) + '.pth')))

        # if useTest and epoch % test_interval == (test_interval - 1):
        if epoch % 5 == 0 and epoch // 10 > 0:
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0
            y_true, y_pred = [], []
            for inputs, labels, filename in tqdm(test_dataloader):
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                inputs = inputs.cuda()
                labels = labels.cuda()

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.long())

                for i in labels.data.cpu().numpy():
                    y_true.append(i)  # 0318
                for i in preds.data.cpu().numpy():
                    y_pred.append(i)

                running_loss += loss.item() * inputs.size(0)
                try:
                    running_corrects += torch.sum(preds == labels.data)
                except:
                    running_corrects += torch.sum(preds.to(torch.int32) == labels)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()  # 0318
            tpr = tp / (tp + fn)  # 0318
            tnr = tn / (tn + fp)
            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)
            writer.add_scalar('data/test_tpr', tpr, epoch)
            writer.add_scalar('data/test_tnr', tnr, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
            with open(txt_file, 'a') as f:
                f.writelines('test_acc:{:.4f}\n'.format(epoch_acc))
                f.writelines('test_tpr:{:.4f}\n'.format(tpr))
                f.writelines('test_tnr:{:.4f}\n'.format(tnr))
            f.close()
    writer.close()


if __name__ == "__main__":
    train_model()
