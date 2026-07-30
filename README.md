# STFT PCG Denoiser — 扩展数据集 vs 原有数据集

一个问题，一条实验：**把 CirCor 的干净心音加进训练集，STFT 去噪模型在自己的设备域上会不会变好。**

模型、损失、超参、验证集、随机种子在两条臂之间完全一致，唯一变化的是训练用的干净心音来源：

| | 训练集（干净心音） | 训练集（噪声） | 验证集 |
|---|---|---|---|
| **臂 A** 原有 | device，3936 窗 | device | device 留出受试者 |
| **臂 B** 扩展 | device + CirCor，约 15000 窗 | device | **同一个** device 留出受试者 |

验证集在两条臂里都是 device-only 且用同一个 seed，所以两个模型看到的是**逐样本完全相同**的混合。
比较因此是配对的 —— 报告的是 per-sample 差值和置信区间，而不是两个独立均值。

---

## 代码出处

| 来源 | 文件 |
|---|---|
| [jiayimaggieshao/denoise_stft](https://github.com/jiayimaggieshao/denoise_stft) | `src/frequency_model.py`、`src/frequency_loss.py`、`src/frequency_metrics.py`、`src/frequency_data.py`、`tests/test_frequency_pipeline.py`、`scripts/check_frequency_pipeline.py`、`data/` 下的 device npz |
| 本项目上游仓库 `Mic_denoise` | `src/circor.py`（CirCor 采样规则） |
| 本文件夹新增 | `src/pools.py`、`scripts/compare_datasets.py`、`scripts/audit_pools.py`、`scripts/build_circor_pool.py`、`scripts/fetch_device_data.py`、`tests/test_pools.py`、`notebooks/`、文档 |

`src/frequency_data.py` 在原版基础上改了三处（尺度归一后的传感器底噪、可调的 chest-only 抖动、
可接收预构建的窗口池），`train_frequency.py` 改成了两臂开关。原始 CirCor 数据来自
[PhysioNet CirCor DigiScope 1.0.3](https://physionet.org/content/circor-heart-sound/1.0.3/)。

## 为什么这个对比不 trivial

CirCor 和 device 数据差得很远，所以「多加数据就会变好」在这里不是显然的：

- **设备不同**：CirCor 是数字听诊器，device 是胸壁贴片麦克风。
- **人群不同**：CirCor 是儿科（0–21 岁，心率中位约 107 bpm），device 是成人（61–79 bpm）。
- **电平差 30–150 倍**：device clean 的 per-window RMS 中位数是 0.00072，CirCor 是 0.02–0.04。

第三点是纯工程问题，`src/pools.py` 把两边归一到同一 RMS 解决了。前两点是真实的域差异 ——
而且因为心音和运动伪影在频谱上几乎完全重叠，模型能依赖的主要线索是**节律**，
心率差在这里不是化妆品级的差别。这也正是这个实验值得跑而不是靠猜的原因。

## 方法（简述）

复数 STFT 域的 2D U-Net + 瓶颈处双向 GRU，用外置麦克风作为条件。
4 kHz、2 s 窗；STFT `n_fft=512 / win=256 / hop=64`，15–800 Hz 软带通，
只把 ≤1000 Hz 的 129 频点 × 126 帧交给网络。

八个输入平面：胸壁复数谱的压缩实/虚部、胸壁与参考的 log 幅度、逐点 SNR 代理、
逐频点参考相干代理、解析先验、参考可用性标志。

关键的一点是网络**不从零学掩膜**：先在 850–1000 Hz（心音通带之外，理论上只有噪声）
估出胸壁/参考的传递幅度比，得到一个解析的参考谱减增益；输出头零初始化，
所以训练起点的掩膜恰好等于这个先验，网络学的是有界的幅度和相位修正
（幅度 logit 域 ±0.75·tanh，相位 ±0.35 rad，掩膜上限 1.2）。

损失 8 项：波形 L1、频率加权的复数 STFT L1、log 幅度、一阶差分、负 SNR、
1−相关系数、五个临床频带的能量比、以及**过度衰减的单边惩罚**。最后两项针对去噪模型
最常见的失败模式：把什么都衰减掉，指标不难看但心音也没了。

训练用的混合在线合成、不落盘：胸壁通道是干净心音加按抽到的 SNR 缩放的噪声，
参考通道是**另一个**噪声窗过随机传递路径（±50 ms 分数延迟、32 阶随机 FIR、随机 EQ、
随机极性、0.5–2 增益），另掺最多 8% 心音泄漏和 −40 dB 传感器底噪，并以 20% 概率整窗丢弃。

## 目录

```
stft/
├── train_frequency.py            训练入口，--no_circor 切换臂 A / 臂 B
├── src/
│   ├── frequency_model.py        CardioSpecNet：复数 STFT 2D U-Net + TF-Grid GRU
│   ├── frequency_loss.py         8 项损失
│   ├── frequency_metrics.py      SNR / SI-SDR / 相关系数 / LSD
│   ├── frequency_data.py         在线合成混合
│   ├── pools.py                  窗口池：格式统一 / 纯净过滤 / RMS 归一 / 来源追溯
│   └── circor.py                 CirCor 采样规则
├── scripts/
│   ├── build_circor_pool.py      CirCor → data/circor/*.npz（跑一次）
│   ├── audit_pools.py            数据体检：格式 / 纯净度 / 尺度 / 划分泄漏
│   ├── compare_datasets.py       两臂在同一评测集上的配对比较 ← 实验结论在这里
│   ├── fetch_device_data.py      重新拉 device npz
│   └── check_frequency_pipeline.py  单测 + 一次真实前向
├── tests/
├── notebooks/stft_dataset_ablation_colab.ipynb
└── data/                         见 data/README.md
```

## 跑法

```bash
pip install -r requirements.txt
```

**1. 建 CirCor 池**（一次性，449 MB 下载）

```bash
wget -O circor.zip https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/
unzip -q circor.zip
PYTHONPATH=. python scripts/build_circor_pool.py --root circor-heart-sound-1.0.3
```

默认取全部合格受试者（丢掉 119 位 `Murmur=Unknown` 的），每人 12 窗，
只在 TSV 非零 state 的连续标注段内采样 —— state 0 是 CirCor 自带的信号质量标签，
那些段里官方明确列了听诊器摩擦、说话、小孩哭笑。细节见 `data/README.md`。

**2. 体检两条臂**

```bash
PYTHONPATH=. python scripts/audit_pools.py --no_circor      # 臂 A
PYTHONPATH=. python scripts/audit_pools.py                  # 臂 B
```

确认格式、纯净度、归一化前的尺度差、以及 train/val 没有受试者泄漏。硬检查失败会 exit 1。

**3. 跑两条臂**（除 `--no_circor` 和输出目录外，命令必须完全一致）

```bash
PYTHONPATH=. python train_frequency.py --smoke --device cpu      # 先跑通

PYTHONPATH=. python train_frequency.py --no_circor \
    --epochs 60 --batch_size 16 --seed 2026 \
    --output_dir checkpoints/device_only

PYTHONPATH=. python train_frequency.py \
    --epochs 60 --batch_size 16 --seed 2026 \
    --output_dir checkpoints/combined
```

每个输出目录里有 `best.pt`、`last.pt`、`history.jsonl`（每 epoch 全部指标）、
`pools.json`（这次实际用了哪些窗口、丢了多少、来源各占多少）。

**4. 比较**

```bash
PYTHONPATH=. python scripts/compare_datasets.py \
    --device_only checkpoints/device_only/best.pt \
    --combined    checkpoints/combined/best.pt
```

评测集在这个脚本里现场构建，不读任何一个训练目录：device 留出受试者的干净心音，
配 device 的 val 噪声，在 −10…+20 dB 七个**固定** SNR 档上各 200 个样本，一个固定 seed。
两个 checkpoint 吃到逐字节相同的混合。

输出每档的 SNRi / SI-SDRi、配对差值 `B − A` 及其 95% 置信区间和配对 p 值，
外加相关系数和对数谱距离，写 `outputs/comparison/{comparison.json,comparison.png}`。

Colab 上一串跑完用 `notebooks/stft_dataset_ablation_colab.ipynb`。

## 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--no_circor` | 关 | 臂 A：只用 device 干净心音训练 |
| `--val_circor` | 关 | 把 CirCor 也放进验证集。**打开后两条臂的评测集就不同了，对比失效** |
| `--circor_pool` | `data/circor/circor_pool_4khz_2s.npz` | 找不到会报错并给出构建命令 |
| `--circor_heart_rate_max` | 无 | 对已建好的池按 bpm 过滤，用来单独检验心率差这一条 |
| `--pool_target_rms` | `0.02` | 所有池归一到的 per-window RMS，`0` 关闭 |
| `--snr_min_db` / `--snr_max_db` | `-10` / `20` | 训练混合的 SNR 抽样范围 |
| `--reference_dropout` | `0.20` | 训练时整窗丢弃参考通道的概率 |
| `--seed` | `2026` | 两条臂必须一致 |

## 结论该怎么读

`compare_datasets.py` 给的是**这两个 checkpoint 在这一个评测集上**的差异，配对且带区间。
它不包含训练种子的方差 —— 两次训练是一个有噪声过程的两个样本。

所以：**如果差值落在置信区间里、或者量级和区间宽度可比，那就还不能算证据**（不管正负）。
换 `--seed` 把两条臂各重跑几次，看差值是否稳定，才是这个实验真正的终点。

另外两条边界：

- 指标全部来自**合成混合**。真实走动录音没有干净参考，做不出数字，所以不在这个实验里。
- 验证集只有一位留出受试者（device 的 4 号，1024 窗）。这是现有数据的上限，
  也意味着「泛化到新受试者」这件事只有 n=1 的证据。
