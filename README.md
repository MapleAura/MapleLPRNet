# MapleLPRNet —— 中文车牌识别 (LPR)

基于 **LPRNetV2** 的中国车牌识别项目，支持端到端训练、验证、单图推理，以及
导出 ONNX（含 RKNN 转换脚本）。模型以 CTC 序列建模，对中文 + 数字 + 英文字母车牌
（支持新能源、普通蓝牌等）进行不定长字符识别，无需字符级分割标注。

## 特性

- **端到端识别**：输入 128x48  车牌子图，直接输出车牌字符串。
- **LPRNetV2 主干**：多尺度特征 + 可配置 head（`grid_h × grid_w` 网格 + CTC）。
- **通道宽度可缩放**：`--width-mult` 一键切换模型大小（约 0.10M ~ 1.5M 参数），
  并支持从已有 checkpoint 做 **Net2Net** 扩/缩权初始化。
- **连号车牌专项处理**：针对 `皖A378663` 这类相邻重复字符，可通过增大
  `--grid-w` 帧数 + 训练侧重复字符加权，显著改善"吞字"问题。
- **难例加权**：用验证集错例清单对新一轮训练加权，损失量级不变、无需重调学习率。
- **完善的落盘记录**：每次训练自动保存 `args.json` / `results.csv` / `train.log`，
  可复现、可直接画曲线。
- **开源模型导出**：checkpoint → ONNX（含 onnxruntime uint8 推理版本）→ RKNN。

## 环境依赖

项目使用 PyTorch，核心依赖如下（建议 `Python 3.8+` + 有 GPU 的机器）：

```
torch>=1.8
torchvision
torch-tb-profiler       # tensorboard 曲线 (from torch.utils.tensorboard)
opencv-python
numpy
tqdm
onnx                    # 导出与验证 ONNX 时使用
onnxruntime             # --onnxruntime 导出(可选)
pandas                  # 只读 results.csv 画曲线时使用(可选)
```

如需在 RKNPU 板端部署，运行时依赖 `rknn-toolkit` 与对应芯片的 `rknn` 库。

## 数据集格式

`--train_img_dirs` 与 `--test_img_dirs` 指向 **txt 列表文件**（多个文件用逗号分隔）。
每行格式为：

```
<图片绝对或相对路径> <车牌字符串> [车牌颜色]
```

其中：

- `<图片路径>` 与 `<车牌字符串>` 按空格分隔，构成 `路径#车牌`。
- 第三个字段（如果存在）是车牌颜色，只有命中白名单
  `["新能源小型车", "普通蓝牌", "黑色车牌"]` 的行才会被读取；不加第三列则全部读入。
- 车牌中的 `I` / `O` 会在加载时自动映射为 `1` / `0`。

示例 `data/train_list.txt`：

```
/home/user/data/1234567.jpg 皖A12345 普通蓝牌
/home/user/data/2345678.jpg 京B88D88 新能源小型车
```

支持在训练时用 `--hard-list` 传入错例清单做难例加权（详见下文）。

## 快速使用

### 1. 训练

```bash
python train.py \
  --train_img_dirs data/train_list.txt \
  --test_img_dirs  data/test_list.txt \
  --epochs 100 \
  --batch-size 128 \
  --device 0 \
  --img-size 128,48
```

训练过程：

- 结果输出在 `runs/expN/`（`N` 自动递增），内含 `args.json`、`results.csv`、
  `train.log`、`weights/`（`best.pt` / `last.pt`）和 TensorBoard 事件。
- 每个 epoch 会在验证集上跑 `--test-epochs`（默认 1）次评测，输出 macc /
  `Repeat subset` 指标。
- 用 TensorBoard 观察曲线：
  ```bash
  tensorboard --logdir runs
  ```

常用辅助参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--resume <ckpt>` | 空 | 从 checkpoint 继续训练（恢复 epoch / 优化器，要求 `width_mult` 一致） |
| `--weights <ckpt>` | 空 | 用预训练权重**初始化**后从 epoch 1 重训（支持宽度变化，自动 Net2Net） |
| `--img-size` | `128,48` | 输入尺寸，支持 `160,48` |
| `--grid-w` | `27` | head 网格宽 = CTC 时间步 T，越长越能解出重复字符 |
| `--width-mult` | `1.0` | 通道宽度系数，<1 变窄、>1 变宽 |
| `--adam` | 关 | 使用 Adam 优化器（默认 SGD+余弦退火） |
| `--repeat-weight` | `0.0` | 相邻重复字符样本加权，建议 0.5~2 |
| `--test-epochs` / `--notest` | `1` / 关 | 控制验证频率，默认每 epoch 都验证 |
| `--nosave` | 关 | 只保存最后一个 checkpoint |

### 2. 测试 / 评估

```bash
python test.py \
  --test_img_dirs data/test_list.txt \
  --weights runs/exp0/weights/best.pt \
  --batch-size 128
```

追加参数：

- `--cpu`：强制 CPU 推理。
- `--float-test`：用 float 精度跑验证（默认 GPU 上自动用 half）。
- `--dump-errors <path>`：把错例写成 `<图片路径> <真值> <预测>`，可直接当
  `--hard-list` 用。
- 输出包含整体 macc，以及专门统计相邻重复字符样本的 `Repeat subset acc / deleted`。

### 3. 单张 / 批量推理

```bash
python detect.py \
  --source-dir /path/to/images \
  --weights runs/exp0/weights/best.pt
```

`--source-dir` 目录下所有图片都会被读取并打印识别结果：

```
1234567.jpg ------> ['皖', 'A', '1', '2', '3', '4', '5']
```

### 4. 导出

**ONNX**（普通浮点，NCHW float 输入）：

```bash
python export.py --weights runs/exp0/weights/best.pt --img-size 128,48
```

**ONNX（onnxruntime / uint8 输入）**：把归一化内嵌进模型，输入为 uint8 NHWC 原图：

```bash
python export.py --weights runs/exp0/weights/best.pt --img-size 128,48 --onnxruntime
```

**RKNN**（NPU 板端）：用 `rknn_export.py` 把导出的 ONNX 量化为 `.rknn`。脚本末尾
需按模型尺寸修三个变量：`ONNX_MODEL`、`RKNN_MODEL`、`DATASET`，以及
`model_h` / `model_w` 注释说明的模型输入尺寸。若模型使用 `head_norm='rms'`
（逐图能量归一化），部分量化工具链不支持，训练时建议改用默认的 `head_norm='bn'`。

---

## 进阶：连号车牌与时间步 `grid-w`

greedy CTC 解码会把连续相同的帧合并，因此 `皖A378663` 这类车牌只有在两个 `6`
之间插入了 blank 时才能解出 `66`。长 `L`、相邻重复 `R` 处的车牌至少需要
`L + R` 帧；帧数不足或模型容量偏小时会把重复字符吞掉（预测比标签短）。

帧数由 head 的 `grid_w` 决定，改成更大的值几乎不增加算力（参数量不变，
MACs 只 +0.1% 以内）。128x48 输入下 `grid_w` 最大 27，160x48 下最大 35：

```bash
python train.py --train_img_dirs data/train.txt --test_img_dirs data/test.txt \
  --img-size 128,48 --grid-w 24
python test.py --test_img_dirs data/test.txt --weights runs/exp0/weights/best.pt
```

`grid_w`（以及 head 结构）会随 checkpoint 一起保存，`test.py` / `export.py` /
`detect.py` 会自动按 checkpoint 还原，无需重复传参。CTC 的 `input_lengths` 由
`grid_w` 推导；`--lpr-max-len` 仅作一致性校验，与 `grid_w` 不一致时会告警并忽略。

训练/测试日志中的 `Repeat subset` 指标专门统计含相邻重复字符样本的准确率与
`deleted`（被吞字符）次数，便于判断连号问题到底是容量不足还是帧数不足。

## 进阶：连号 / 难例加权

CTC 下 `11` 这类相邻重复字符必须在两帧之间插入一帧 blank 才能解出，是数据里的
难例。两个开关都只影响训练损失，不改模型结构、不改输入尺寸、不影响导出与推理：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--repeat-weight` | `0.0` | 样本权重 = `1 + α × 相邻重复字符数 R`；0 表示关闭，建议 0.5~2 |
| `--hard-list` | 空 | 难例清单（逗号分隔的 txt），命中的图片额外加权；按**文件名**匹配，不要求路径前缀一致 |
| `--hard-weight` | `2.0` | `--hard-list` 里样本的权重倍数 |

权重在 loss 里按均值归一，所以损失量级不变、学习率不用重调（等价于只改变样本
之间的相对权重，`R=0` 的样本会被相对降权）。不开这两个开关时走原来的
`reduction='mean'` 路径，与改动前逐位一致。

难例清单直接用验证集跑出来的错例即可：

```bash
python test.py --test_img_dirs data/test.txt \
  --weights runs/exp7/weights/best.pt --dump-errors runs/errs.txt

python train.py --train_img_dirs data/train.txt --test_img_dirs data/test.txt \
  --weights runs/exp7/weights/best.pt --repeat-weight 1.0 \
  --hard-list runs/errs.txt --hard-weight 2.0
```

每个 epoch 开头的 `Repeat stats` 会打印连号样本占比与权重分布；是否有效看
`test.py` 的 `Repeat subset acc / deleted` 两个指标。

## 进阶：用 `--width-mult` 缩放通道宽度

`width_mult` 同时缩放 backbone 的 c1/c2/c3（64/128/256），可直接取小于 1 的
值，例如 `--width-mult 0.5` 得到 c1/c2/c3 = 32/64/128：

| width_mult | c1/c2/c3 | Params | MACs (128x48) |
|---|---|---|---|
| 1.0 | 64/128/256 | 0.70M | 965M |
| 0.5 | 32/64/128 | 0.25M | 291M |
| 0.25 | 16/32/64 | 0.10M | 115M |

**从零训练**：直接给 `--width-mult 0.5` 即可，无需其它参数。

**从已有更宽的 checkpoint 初始化**（`--weights`）：会自动做 Net2Net 缩权 —— 按
通道重要性（有 BN 时用 BN 的 gamma，否则用卷积核 L2 能量）选出子网络，把权重裁剪
进窄模型。缩权不是精确等价的（被裁掉的通道信息会丢失），只是更好的初始化，
因此必须重新训练，不能用 `--resume`。缩权后各层激活分布改变，checkpoint 里的
BN running 统计不再匹配，训练脚本会用 `--bn-recalib-batches`（默认 20）个 batch
重新估计 BatchNorm 统计，否则初始 loss 会被 BN 放大若干倍。

```bash
python train.py --train_img_dirs data/train.txt --test_img_dirs data/test.txt \
  --img-size 128,48 \
  --weights runs/exp3/weights/best.pt --width-mult 0.5 --grid-w 24
```

同理，从较窄的 checkpoint 用更大的 `--width-mult` 初始化时会自动做 Net2Net **扩权**
（复制 + 通道扩展），适合从小模型起步逐步增大容量。

## 训练落盘记录

每次训练都会在本次运行目录（`runs/expN/`）下留下三份记录，事后复现和画曲线都
用它们：

| 文件 | 内容 |
|---|---|
| `args.json` | 本次训练用的全部配置（命令行参数 + head_cfg + 时间步 T + 参数量 + 数据集大小 + 时间/主机） |
| `results.csv` | 每个 epoch 一行：`epoch, lr, train_loss, train_loss_weighted, val_loss, macc, repeat_n, repeat_correct, repeat_deleted, best_acc, is_best, epoch_time_s, total_time_s` |
| `train.log` | 控制台日志（带时间戳），与 `--test-epochs`/`--notest` 无关，全程都记 |

没做验证的 epoch（受 `--test-epochs` / `--notest` 控制）验证列留空，行照样写，
所以 `results.csv` 的 `epoch` 列永远是连续的。CSV 每行写完立即 flush，训练中断也
不会丢已完成的 epoch。

```python
import pandas as pd
df = pd.read_csv('runs/exp0/results.csv')   # 直接画 macc / repeat_deleted 曲线
```

## 常用模型参数速查

head 相关参数（`--head-ch` / `--head-ksize` / `--head-pool` / `--head-norm` /
`--grid-h` / `--grid-w`）会随 checkpoint 保存并在推理/导出时自动还原，训练时一次
敲定即可，无需在 `test.py` / `detect.py` / `export.py` 重复传参。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--grid-h` | `4` | head 网格高度（沿时间轴平均池化） |
| `--head-ch` | `None` | 每分支先做卷积压缩到此通道数；`None` 表示只池化不卷积 |
| `--head-ksize` | `1` | head 卷积核：`1`=逐点，`3/5`=可学习空间卷积（需配合 `--head-ch`） |
| `--head-norm` | `bn` | 分支归一化：`bn`（可部署）、`rms`（原版逐图能量归一化，精度高但部分量化链路不支持）、`none` |
| `--head-pool` | `avg` | 网格下采样池化：`avg` / `max` |
| `--bn-recalib-batches` | `20` | 缩权初始化后重估 BatchNorm 的 batch 数，`0` 关闭 |

## 目录结构

```
MapleLPRNet/
├── train.py            # 训练入口
├── test.py             # 验证 / 导出错例
├── detect.py           # 单目录批量推理
├── export.py           # checkpoint → ONNX
├── rknn_export.py      # ONNX → RKNN (NPU 板端)
├── model/
│   ├── lprnet.py       # LPRNetV2 模型 + CHARS 字符表
│   ├── stnet.py        # 空间变换网络子模块
│   └── plateNet.py
├── data/
│   ├── dataset.py      # 数据集加载 + 训练时色彩增强
│   └── bases.py
├── utils/
│   └── general.py      # CTC 解码 / Net2Net 扩缩权 / BN 校准等工具
└── runs/               # 训练输出（exp0, exp1, ...）
```
