# 两点车轨耦合数据生成

本项目生成静止磁浮悬浮单元在同一柔性梁上的前后两点耦合振动数据，用于数据驱动预测方法验证。车辆保留两电磁铁与共同上部质量的三个机械自由度，梁保留第一阶或前两阶模态；控制采用间隙比例、磁铁绝对速度反馈和完整电流回路。

模型推导见 [dynamic_model.md](docs/dynamic_model.md)，文件职责和数据合同见 [project.md](docs/project.md)。坐标向下为正，梁坐标沿前进方向增大；点1为后点，点2为前点。

## 1. 环境与生成

Python 3.11+。在项目根目录运行：

```bash
python -m pip install -r requirements.txt
python make_data.py default.yaml
```

入口接收 `configs` 下的一个 YAML 文件名。复制配置为 `configs/my_data.yaml` 后可运行 `python make_data.py my_data.yaml`。数据写入 `data/<dataset.name>.npz`，开启作图时同时生成同名 PNG；终端逐条显示有效采样数和终止原因。同名 NPZ 在整批生成后原子替换。

默认生成4条、每条10001点，采样间隔1 ms，时间为0—10 s。默认无外力，初始位移和模态位移独立采样10 μm标准差的正态微扰，初速度与电流增量为零。

## 2. 配置

所有数值参数和运行设置均来自 [configs/default.yaml](configs/default.yaml)。

| 要调整的内容 | 配置项 | 含义 |
| --- | --- | --- |
| 名称与规模 | `dataset.name`、`dataset.n_samples` | 输出名称、预定轨迹数 |
| 公共时间轴 | `time.n_steps`、`time.dt` | `t=dt*arange(n_steps)`，时长为 `(n_steps-1)*dt` |
| 模态数 | `beam.n_modes` | 1或2；完整状态分别为10或12维 |
| 梁与位置 | `beam`、`geometry.positions_m` | 跨度、线密度、一阶频率、阻尼及后/前位置 |
| 悬浮与连接件 | `vehicle`、`magnet`、`controller` | 质量、刚度、阻尼、电磁参数、物理边界和增益 |
| 初值 | `initial.offsets`、`initial.random_std` | 状态名到增量/正态标准差的映射 |
| 外力 | `excitation` | `none`或两悬浮位置的梁上有限时长`pulse` |
| 随机性 | `dataset.seed`、`dataset.split_seed` | 参数/初值种子、划分种子 |
| 参数变化 | `dataset.parameter_ranges` | 点分路径到 `[下界,上界]` 的均匀采样范围 |
| 参数族 | `dataset.n_parameter_families`、`dataset.split_by` | 共享参数的族数；按`trajectory`或`family`划分 |
| 精度 | `integrator`、`dataset.dtype` | Radau容差、最大步长；float32/float64存储 |
| 验收 | `validation` | 启用模型、积分、数据与完整科学验收 |

初值键使用状态名，如`z_m1`、`z_m2`、`z_c`、`v_m1`、`q1`、`i1`，各项均为相对平衡状态的增量。`common_m`给两磁铁增加相同位移，`differential_m`给后点增加、前点减去相同位移，并与逐状态项叠加。一阶配置忽略`random_std`中的`q2`、`p2`；`offsets`使用当前状态名或共同/差动键。

脉冲为 `amplitudes_n * sin(pi*(t-start_s)/duration_s)^2`，仅在起止时间之间生效。两个通道按后点、前点排列。参数范围可用`controller.current_gain_v_a`、`vehicle.m1_kg`、`geometry.positions_m.0`等路径；连续参数均匀采样，线圈匝数在区间内的整数中均匀采样。每份文件固定模态数与公共时间轴。

## 3. 读取与使用

```python
import numpy as np
from sim.dataset import load_dataset

d = load_dataset("data/stationary_two_point.npz")
t, Y, U = d["t"], d["Y"], d["U"]
print(Y.shape, d["Y_names"])
train = np.flatnonzero(d["split"] == "train")
# 先选择整条轨迹的split，再在valid连续有效段切训练窗口。
```

也可用`np.load(path, allow_pickle=False)`直接读取。

| 字段 | 形状 | 列序与内容 |
| --- | --- | --- |
| `t` | `(T,)` | 公共时间轴，s |
| `Y` | `(N,T,5)` | `[z_c,z_m1,z_m2,w1,w2]`，平衡增量位移，m |
| `U` | `(N,T,2)` | `[P1,P2]`，梁上外力，N |
| `X` | `(N,T,10/12)` | `[z_m1,z_m2,z_c,v_m1,v_m2,v_c,q1,…,qJ,p1,…,pJ,i1,i2]` |
| `O` | `(N,T,8)` | `[gap1,gap2,a_m1,a_m2,a_c,i1,i2,beam_port_power]` |
| `valid` | `(N,T)` | 有效采样前缀；失败尾段四组张量均填NaN |
| `split` | `(N,)` | 整轨迹或整参数族的`train / val / test` |

间隙和电流保存实际值，梁端口功率为电磁增量力向梁输入的功率。列名和单位由`*_names`与`*_units`给出。每条预定轨迹均有记录，包括初始越界、物理事件和积分失败；终止时间与完整终止状态独立保存。实际参数、初值、平衡量、种子及来源校验和随文件保存，详见架构文档。

## 4. 起始算例与验证

默认组合参数采用附件机械参数`m1=m2=300 kg`、`m3=330 kg`、`m4=2437.5 kg`，两连接分别为`k=2e7 N/m`、`c=5000 N·s/m`。梁长24 m、线密度400 kg/m，后/前位置为9.6/14.4 m。一阶频率约5.588844661 Hz，第二阶为其4倍。10 mm受载平衡得到每点承载力16517.5875 N、电流约25.04296159 A。参数来源及`m3=330/450 kg`差异保存在配置和模型文档中。

在`kp=5500 A/m`、`kd=50 A·s/m`下，`current_gain_v_a: 65`为增长起始工况，改为80为衰减起始工况。科学验收计算刚轨、一阶与两阶特征值、时域增长率和频率、对称/反对称响应、能量功率及步长收敛；这组组合参数的验收结果见模型文档。

默认开启模型、积分和数据合同检查。将`validation.scientific`设为`true`可运行完整科学验收；结果同时写入NPZ的`validation_json`和同名`_validation.json`。验收使用`validation`中记录的时长、拟合区间、增益与种子，并保存各对照的实际配置及数值差值。

v2样例、原附件基线与科学验收记录位于`data/example`。旧单点v1样例位于`data/legacy_v1`；已有本地`data/due_v2_benchmark`、`data/due_large_10000`也属于旧单点合同，名称中的v2不表示本项目的新格式。新格式以`format_version=2`、`model_version=stationary_two_point_v2`和五列Y识别。
