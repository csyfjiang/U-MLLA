import torch
import copy
from mlla_unet_old_merging import MLLA_UNet
from torchsummary import summary

def print_model_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")


def load_from(model, pretrained_path):
    if pretrained_path is not None:
        print(f"pretrained_path: {pretrained_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)

        if "model" not in pretrained_dict:
            print("---start load pretrained model by splitting---")
            pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
        else:
            print("---start load pretrained model of swin encoder---")
            pretrained_dict = pretrained_dict['model']

        model_dict = model.state_dict()
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
        msg = model.load_state_dict(model_dict, strict=False)
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")

        return not_loaded_keys, missing_keys
    else:
        print("No pretrained weights")
        return [], []


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = MLLA_UNet(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=64,
        depths=[2, 4, 8, 4],
        depths_decoder=[1, 2, 2, 2],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        drop_path_rate=0.1,
        ape=False,
        use_checkpoint=False
    ).to(device)
    # 生成随机输入张量
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

    # 打印模型摘要
    summary(model, (3, 224, 224))

    # 运行模型
    print("\nRunning model with random input...")
    with torch.no_grad():
        output = model(input_tensor)

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # # 打印模型结构
    # print("Initial model weights:")
    # print_model_weights(model)
    #
    # # 打印总参数数量
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"\nTotal trainable parameters: {total_params}")
    #
    # # 加载预训练权重
    # checkpoint_path = '../pretrained_ckpt/MLLA-T.pth'
    # not_loaded_keys, missing_keys = load_from(model, checkpoint_path)
    #
    # print("\nWeights after loading checkpoint:")
    # print_model_weights(model)
    #
    # print("\nKeys in checkpoint that were not loaded into the model:")
    # for key in not_loaded_keys:
    #     print(key)
    #
    # print("\nKeys in model that were not found in the checkpoint:")
    # for key in missing_keys:
    #     print(key)