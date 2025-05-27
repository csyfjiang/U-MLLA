import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
try:
    from datasets.dataset_acdc import BaseDataSets as ACDC_dataset
except:
    pass
from datasets.dataset_ab import dataset_ab
from utils import test_single_volume
# from networks.vision_transformer import SwinUnet as ViT_seg
from networks.MLLA_Unet_Build import MLLAUnet as ViT_seg
from trainer import trainer_synapse
from config_mlla_unet import get_config

device = torch.device("cuda:1")

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device to use for computation: cpu, cuda:0, cuda:1, ..., mps')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)


def inference(args, model, test_save_path=None,dataset_name=None):
    # txt命名在这里
    db_test = args.Dataset(base_dir=args.volume_path, split="val", list_dir=args.list_dir, dataset_name = dataset_name)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    device = torch.device(args.device)
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, device=device, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    # Validate and set the device
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this system.")
        device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid CUDA device ID: {device_id}. Available devices: {torch.cuda.device_count()}")
    elif args.device == 'mps':
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is not available on this system.")
    elif args.device != 'cpu':
        raise ValueError(f"Invalid device specified: {args.device}")
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    dataset_config = {
        'ACDC': {
            'Dataset': ACDC_dataset,  # datasets.dataset_acdc.BaseDataSets,
            'volume_path': './data/ACDC',
            'list_dir': None,
            'num_classes': 4,
            'z_spacing': 5,
            'info': '3D'
        },
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
        'flare22': {
            'Dataset': dataset_ab,
            'volume_path': './data/data',
            'num_classes': 14,
            'list_dir': './lists/lists_flare22',
            'z_spacing': 2.5,
        },
        'altas': {
            'Dataset': dataset_ab,
            'volume_path': './data/data',
            'num_classes': 3,
            'list_dir': './lists/lists_altas',
            'z_spacing': 1,
        },
        'amos': {
            'Dataset': dataset_ab,
            'volume_path': './data/data',
            'num_classes': 16,
            'list_dir': './lists/lists_amos',
            'z_spacing': 5,
        },
        'amos_mr': {
            'Dataset': dataset_ab,
            'volume_path': './data/data',
            'num_classes': 16,
            'list_dir': './lists/lists_amos_mr',
            'z_spacing': 5,
        },
        'word': {
            'Dataset': dataset_ab,
            'volume_path': './data/data',
            'num_classes': 17,
            'list_dir': './lists/lists_word',
            'z_spacing': 2.5,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True
    

    # net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(device)

    # snapshot = os.path.join(args.output_dir, 'best_model.pth')
    # # if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    # msg = net.load_state_dict(torch.load(snapshot))
    # print("self trained swin unet",msg)
    # snapshot_name = snapshot.split('/')[-1]

    # log_folder = './test_log/test_log_'
    # os.makedirs(log_folder, exist_ok=True)
    # logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    # logging.info(snapshot_name)

    # if args.is_savenii:
    #     args.test_save_dir = os.path.join(args.output_dir, "predictions")
    #     test_save_path = args.test_save_dir 
    #     os.makedirs(test_save_path, exist_ok=True)
    # else:
    #     test_save_path = None
    # inference(args, net, test_save_path)
    # New ↓ Test multiple model ckpts


    model_list = [f'best_model_val_{dataset_name}.pth', f'best_model_{dataset_name}.pth', f'final_model_epoch_612_{dataset_name}.pth']
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(device)

    for model_name in tqdm(model_list, desc="Testing models"):
        snapshot = os.path.join(args.output_dir, model_name)
        msg = net.load_state_dict(torch.load(snapshot))
        print(f"Loaded model: {model_name}, Message: {msg}")
        
        log_folder = f'./test_log/test_log_{dataset_name}'
        os.makedirs(log_folder, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_folder, f'{model_name}.txt'), 
                            level=logging.INFO, 
                            format='[%(asctime)s.%(msecs)03d] %(message)s', 
                            datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        logging.info(model_name)

        if args.is_savenii:
            args.test_save_dir = os.path.join(args.output_dir, f"predictions_{dataset_name}_{model_name}")
            test_save_path = args.test_save_dir 
            os.makedirs(test_save_path, exist_ok=True)
        else:
            test_save_path = None
        print(device)
        inference(args, net, test_save_path, dataset_name=dataset_name)

        # 清除之前的日志处理器，避免重复日志
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

