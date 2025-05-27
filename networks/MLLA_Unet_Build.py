# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import torch
import torch.nn as nn
from .mlla_unet import MLLA_UNet

logger = logging.getLogger(__name__)

class MLLAUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(MLLAUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.mlla_unet = MLLA_UNet(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.MLLA.PATCH_SIZE,
            in_chans=config.MODEL.MLLA.IN_CHANS,
            num_classes=self.num_classes,
            embed_dim=config.MODEL.MLLA.EMBED_DIM,
            depths=config.MODEL.MLLA.DEPTHS,
            depths_decoder=config.MODEL.MLLA.DEPTHS_DECODER,
            num_heads=config.MODEL.MLLA.NUM_HEADS,
            mlp_ratio=config.MODEL.MLLA.MLP_RATIO,
            qkv_bias=config.MODEL.MLLA.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.MLLA.APE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.mlla_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print(f"pretrained_path: {pretrained_path}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)

            if "model" not in pretrained_dict:
                print("---start load pretrained model by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
            else:
                print("---start load pretrained model of mlla encoder---")
                pretrained_dict = pretrained_dict['model']

            model_dict = self.mlla_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)

            # Copy weights from 'layers' to 'layers_up'
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = f"layers_up.{current_layer_num}{k[8:]}"
                    if current_k in model_dict:
                        if v.shape == model_dict[current_k].shape:
                            full_dict[current_k] = v
                        else:
                            print(f"Shape mismatch for {current_k}, skipping")

            # Handle special cases for layers_up
            for k in model_dict.keys():
                if k.startswith("layers_up") and k not in full_dict:
                    if k.replace("layers_up", "layers") in pretrained_dict:
                        full_dict[k] = pretrained_dict[k.replace("layers_up", "layers")]

            # Delete weights that don't match in shape or are in the output layer
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print(f"delete: {k}; shape pretrain: {full_dict[k].shape}; shape model: {model_dict[k].shape}")
                        del full_dict[k]
                elif "output" in k or "head" in k:
                    print(f"delete key: {k}")
                    del full_dict[k]

            not_loaded_keys = [k for k in full_dict.keys() if k not in model_dict]
            missing_keys = [k for k in model_dict.keys() if k not in full_dict]

            model_dict.update(full_dict)
            msg = self.mlla_unet.load_state_dict(model_dict, strict=False)
            print(f"Missing keys: {msg.missing_keys}")
            print(f"Unexpected keys: {msg.unexpected_keys}")

            return not_loaded_keys, missing_keys
        else:
            print("No pretrained weights")
            return [], []

# 示例用法
if __name__ == "__main__":

    import argparse
    from config_mlla_unet import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='../configs/mlla_t.yaml', type=str, help='path to config file')
    parser.add_argument('--pretrained', default='../pretrained_ckpt/MLLA-T.pth', type=str, help='path to pretrained model')
    parser.add_argument('--root_path', type=str,
                        default='data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=9, help='output channel of network')
    parser.add_argument('--output_dir', type=str, help='output dir')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
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
    args = parser.parse_args()

    config = get_config(args)

    model = MLLAUnet(config)
    if args.pretrained:
        model.load_from(config)

    # 示例输入
    x = torch.randn(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
    output = model(x)
    print(f"Output shape: {output.shape}")