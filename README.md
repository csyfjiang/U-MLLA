# U-MLLA

The codes for the work "U-MLLA: A Cognitive-Inspired Enhancement of Linear Attention for Medical Image Segmentation". 

## 1. Pretrained Models: You can choose a pretrained model based on your preference.

| model  | Resolution | #Params | FLOPs | acc@1 |            config            |                      pretrained weights                      |
| ------ | :--------: | :-----: | :---: | :---: | :--------------------------: | :----------------------------------------------------------: |
| MLLA-T |    224     |   25M   | 4.2G  | 83.5  | [config](./cfgs/mlla_t.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/7a19712877cb4242889c/?dl=1) |
| MLLA-S |    224     |   43M   | 7.3G  | 84.4  | [config](./cfgs/mlla_s.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/0e5d0b1409d540aaa80c/?dl=1) |
| MLLA-B |    224     |   96M   | 16.2G | 85.3  | [config](./cfgs/mlla_b.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/91c85c5a1061496d8796/?dl=1) |

Ref: [MLLA Official Implementation]

## 2. Prepare data

- [AMOS 22](https://amos22.grand-challenge.org/Dataset/)
- [FLARE 22](https://flare22.grand-challenge.org/)
- [ATLAS](https://atlas.grand-challenge.org/)
- [WORD](https://github.com/HiLab-git/WORD)
- [BTCV](https://www.synapse.org/Synapse:syn3193805/wiki/89480)
- [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

### Preprocessing:

Ref: [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
Detailed procedure: [link](https://github.com/csyfjiang/U-MLLA+/blob/main/datasets/README.md)

Please follow the above procedure from the scratch, you are not recommended use the preprocessed data from the other work directly, otherwise it would get the worse results.

## 3. Environment

- Please prepare an environment with python=3.9 and then use the command `pip install -r requirements.txt` for the dependencies.

## 4. Train/Test

- Run the train script on synapse dataset. The batch size we used is 48. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

- Train

```bash (Recommend)
sh train.sh
```

- Test 

```bash(Recommend)
sh test.sh
```

## References

* [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet)
* [MLLA](https://github.com/LeapLabTHU/MLLA)
