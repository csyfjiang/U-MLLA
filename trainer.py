import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume

from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

def trainer_ab(args, model, snapshot_path):
    dataset_name = args.dataset
    device = torch.device(args.device) 
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[
                            logging.FileHandler(snapshot_path + f"/log_{dataset_name}.txt"),
                            logging.StreamHandler(sys.stdout)
                        ])
    from datasets.dataset_ab import dataset_ab, RandomGenerator
    logging.info("Starting trainer_acdc function")
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    
    db_train = dataset_ab(base_dir=args.root_path, split="train", list_dir=args.list_dir, transform=transforms.Compose([
        RandomGenerator([args.img_size, args.img_size])]),dataset_name=dataset_name)
    db_val = dataset_ab(base_dir=args.root_path, split="val", list_dir=args.list_dir ,dataset_name=dataset_name)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    
    model.train()
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    max_epoch = max_iterations // len(trainloader) + 1
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
    
    ce_loss = CrossEntropyLoss(ignore_index=-1)
    dice_loss = DiceLoss(num_classes, ignore_index=-1)

    writer = SummaryWriter(snapshot_path + f'/log_{dataset_name}')
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    best_performance = 0.0
    best_epoch = 0
    best_loss = float('inf')
    best_dice_loss = float('inf')
    iter_num = 0
    
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        epoch_loss = 0
        epoch_dice_loss = 0
        
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            outputs = model(volume_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)  # 添加 diceloss 的输出
            
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            epoch_loss += loss.item()
            epoch_dice_loss += loss_dice.item()

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num >= max_iterations:
                break

        avg_loss = epoch_loss / len(trainloader)
        avg_dice_loss = epoch_dice_loss / len(trainloader)
        
        # 检查是否是最佳模型
        if avg_loss < best_loss and avg_dice_loss < best_dice_loss:
            
            best_loss = avg_loss
            best_dice_loss = avg_dice_loss
            best_epoch = epoch_num
            best_model_path = os.path.join(snapshot_path, f'best_model_{dataset_name}.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved at epoch {epoch_num} with loss: {avg_loss:.4f} and Dice loss: {avg_dice_loss:.4f}")

        # 每8个epoch进行验证
        if (epoch_num + 1) % 8 == 0:
            model.eval()
            metric_list = 0.0
            for i_batch, sampled_batch in enumerate(valloader):
                image, label = sampled_batch["image"].to(device), sampled_batch["label"].to(device)
                metric_i = test_single_volume(image, label, model, classes=num_classes,
                                              patch_size=[args.img_size, args.img_size], device=device)
                metric_list += np.array(metric_i)
            metric_list = metric_list / len(db_val)
            performance = np.mean(metric_list, axis=0)[0]
            mean_hd95 = np.mean(metric_list, axis=0)[1]

            writer.add_scalar('info/val_mean_dice', performance, iter_num)
            writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

            logging.info(f'Epoch {epoch_num}, Validation Mean Dice: {performance:.4f}, Mean HD95: {mean_hd95:.4f}')

            if performance > best_performance and avg_loss <= best_loss and avg_dice_loss <= best_dice_loss:
                best_performance = performance
                best_epoch = epoch_num
                save_best = os.path.join(snapshot_path, f'best_model_val_{dataset_name}.pth')
                torch.save(model.state_dict(), save_best)
                logging.info(f'Best validation model saved at epoch {epoch_num} with mean_dice: {performance:.4f}, mean_hd95: {mean_hd95:.4f}')
            else:
                logging.info(f'Current best validation model remains at epoch {best_epoch} with mean_dice: {best_performance:.4f}')

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch_num}, current lr {current_lr}')

        if iter_num >= max_iterations:
            break

    save_mode_path = os.path.join(snapshot_path, f'final_model_epoch_{dataset_name}.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info(f"Completed training. Saved final model to {save_mode_path}")
    
    logging.info(f"Best model was saved at epoch {best_epoch} with loss: {best_loss:.4f} and Dice loss: {best_dice_loss:.4f}")
    writer.close()
    return "Training Finished!"


def trainer_acdc(args, model, snapshot_path):
    device = torch.device(args.device)
    
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[
                            logging.FileHandler(snapshot_path + "/log_ACDC.txt"),
                            logging.StreamHandler(sys.stdout)
                        ])
    from datasets.dataset_acdc import BaseDataSets, RandomGenerator
    logging.info("Starting trainer_acdc function")
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator([args.img_size, args.img_size])]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    model.train()
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    # Epoch 逻辑
    max_iterations = args.max_iterations
    max_epoch = max_iterations // len(trainloader) + 1
    
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
    
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log_ACDC')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info("{} val iterations per epoch".format(len(valloader)))
    iter_num = 0
    
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    best_performance = 0.0
    best_epoch = 0
    best_loss = float('inf')
    best_dice_loss = float('inf')
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            outputs = model(volume_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)  # 添加 diceloss 的输出
            
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 500 == 0:  # 500
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    image, label = sampled_batch["image"], sampled_batch["label"]
                    metric_i = test_single_volume(image, label, model, classes=num_classes,
                                                  patch_size=[args.img_size, args.img_size])
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                
                if performance > best_performance:
                    best_iteration, best_performance, best_hd95 = iter_num, performance, mean_hd95
                    best_epoch = epoch_num
                    best_loss = loss.item()
                    best_dice_loss = loss_dice.item()
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_best)
                    logging.info('Best model | iteration %d : mean_dice : %f mean_hd95 : %f' % (
                    iter_num, performance, mean_hd95))

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num >= max_iterations:
                logging.info(f"Reached maximum iterations ({max_iterations}). Stopping training.")
                save_mode_path = os.path.join(snapshot_path, f'final_model_iter_{iter_num}.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info(f"Saved final model to {save_mode_path}")
                break  # 跳出内部循环

        if iter_num >= max_iterations:
            break  # 跳出外部循环
        
        # 在每个 epoch 结束后更新学习率
        scheduler.step()

        # 输出当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch_num}, current lr {current_lr}')

    # 如果达到这里，说明已经完成了所有预定的 epoch 或达到了最大迭代次数
    save_mode_path = os.path.join(snapshot_path, f'final_model_epoch_{epoch_num}.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info(f"Completed training. Saved final model to {save_mode_path}")
    
    logging.info(f"Best model was saved at epoch {best_epoch} with loss: {best_loss:.4f} and Dice loss: {best_dice_loss:.4f}")
    writer.close()
    return "Training Finished!"
    
    
def trainer_synapse(args, model, snapshot_path):
    device = torch.device(args.device)
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    max_epoch = args.max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0


    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    best_loss = float('inf')
    best_dice_loss = float('inf')
    best_epoch = -1
    
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        epoch_loss = 0
        epoch_dice_loss = 0
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            epoch_loss += loss.item()
            epoch_dice_loss += loss_dice.item()
            
        avg_loss = epoch_loss / len(trainloader)
        avg_dice_loss = epoch_dice_loss / len(trainloader)
        
        # 检查是否是最佳模型
        if avg_loss < best_loss and avg_dice_loss < best_dice_loss:
            best_loss = avg_loss
            best_dice_loss = avg_dice_loss
            best_epoch = epoch_num
            best_model_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved at epoch {epoch_num} with loss: {avg_loss:.4f} and Dice loss: {avg_dice_loss:.4f}")
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # 在每个 epoch 结束后更新学习率
        scheduler.step()

        # 输出当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch_num}, current lr {current_lr}')

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
            
    logging.info(f"Best model was saved at epoch {best_epoch} with loss: {best_loss:.4f} and Dice loss: {best_dice_loss:.4f}")
    writer.close()
    return "Training Finished!"
