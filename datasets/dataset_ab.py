import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class dataset_ab(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, dataset_name = None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.dataset_name = dataset_name  # 保存 dataset_name 为实例变量
        
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, 'slice',self.dataset_name, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['data'][0], data['data'][1]
        else:  # Assuming this is for validation or test
            vol_name = self.sample_list[idx].strip('\n')
            filepath = os.path.join(self.data_dir, "val_h5" ,f'val_h5_{self.dataset_name}', f"{vol_name}.h5")
            with h5py.File(filepath, 'r') as data:
                image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

def test_dataset():
    # 设置路径
    base_dir = "../data/data"  # 根据您的实际数据路径进行调整
    list_dir = "../lists/lists_flare22"
    
    # 创建数据集实例
    train_dataset = dataset_ab(base_dir=base_dir, list_dir=list_dir, split="train", 
                               transform=RandomGenerator(output_size=[224, 224]),dataset_name='flare22')
    val_dataset = dataset_ab(base_dir=base_dir, list_dir=list_dir, split="val",dataset_name='flare22')

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 测试训练集
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    train_sample = next(iter(train_loader))
    print("训练样本:")
    print(f"图像形状: {train_sample['image'].shape}")
    print(f"标签形状: {train_sample['label'].shape}")
    print(f"案例名称: {train_sample['case_name']}")

    # 测试验证集
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    val_sample = next(iter(val_loader))
    print("\n验证样本:")
    print(f"图像形状: {val_sample['image'].shape}")
    print(f"标签形状: {val_sample['label'].shape}")
    print(f"案例名称: {val_sample['case_name']}")

    # 可视化一个训练样本
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(train_sample['image'][0, 0].numpy(), cmap='gray')
    plt.title("训练图像")
    plt.subplot(122)
    plt.imshow(train_sample['label'][0].numpy(), cmap='gray')
    plt.title("训练标签")
    plt.show()

# if __name__ == "__main__":
#     pass

#     # test_dataset()


