# 单点车轨耦合动力学生成器

本仓库只生成 **DUE 可直接加载** 的张量，不训练网络、不封装 Gym、不保存仿真内部状态。

- 需求说明：[docs/单点车轨耦合动力学生成器_Python开发需求.md](docs/单点车轨耦合动力学生成器_Python开发需求.md)
- 单位与方程：[docs/MODEL.md](docs/MODEL.md)

默认参数是演示集（`parameter_source: demo_not_hs_maglev`），内部自洽，但不是已验证的高速磁浮算例。

## 架构

```
python -m maglev_sim          # 原有通用生成入口
        │
        ▼
   maglev_sim/due.py          # 写出 Y、U
        │
        ▼
   maglev_sim/plant/          # 内部闭环 ODE，DUE 训练时不要调用
     physics / control / simulate
```

| 层级 | 路径 | 作用 |
|---|---|---|
| 对外 | `maglev_sim/due.py`、`plot.py`、`cli.py` | 生成、读取、画 DUE 张量 |
| 对内 | `maglev_sim/plant/` | 磁浮–梁闭环积分，不进入 NPZ |

原有公开接口为：`generate_due_dataset`、`load_due_dataset`、`plot_due_dataset`、`load_config`。

另有标准化 benchmark 生成入口 `python -m maglev_sim.benchmark`，支持时变输入、
数值收敛检查、ID/OOD 数据划分与断点续跑；`scripts/extend_benchmark.py` 可扩充独立轨迹。
文件职责、生成命令和 DUE 使用方式见 [benchmark 文档](docs/benchmark.md)。
本地 `data/due_v2_benchmark/` 和 `data/due_large_10000/` 使用这套流程；
DUE 训练只需读取生成好的 NPZ，无需调用 benchmark 代码。

## DUE 文件里有什么

一个 `.npz` **只含** 下面 7 个键，没有电流、模态、间隙、加速度或元数据堆砌。

| 键 | 形状 | 含义 |
|---|---|---|
| `Y` | `(N, T, 3)` | 预测目标：`z_car`，`z_magnet`，`beam_displacement_at_magnet` |
| `U` | `(N, T, 2)` | 已知输入：`irregularity`（\(r\)），`force_P`（梁上外力） |
| `t` | `(T,)` | 时间轴 [s] |
| `dt` | 标量 | 采样周期 [s] |
| `split` | `(N,)` | 按**整条轨迹**划分的 `train` / `val` / `test` |
| `Y_names` | `(3,)` | 与 `Y` 最后一维对齐 |
| `U_names` | `(2,)` | 与 `U` 最后一维对齐 |

网络用法：历史段 `Y` + 整段 `U` → 预测未来段 `Y`。`U` 可以含未来（不平顺和外力是已知的）。**不要把未来的 `Y[:,:,2]`（梁位移）当输入。**

```python
from maglev_sim import load_due_dataset

d = load_due_dataset("data/example/due_N100_T1000.npz")
Y, U, t = d["Y"], d["U"], d["t"]     # (100, 1000, 3), (100, 1000, 2), (1000,)
# 例如窗口：用 [:, :800] 的 Y 与全部 U 预测 [:, 800:] 的 Y
```

下面这些量在积分里存在，但**不会**写入文件：速度、加速度、间隙、电流、电磁力、控制器状态、梁模态 \(q_j,p_j\)。间隙 \(\delta=\delta_0+z_m-r-w\)，与 `Y`、`U` 线性相关。

## 安装

```bash
pip install -e ".[dev]"
```

Python 3.11+，NumPy、SciPy、PyYAML、Matplotlib。

## 示例：100 条 × 1000 点

仓库自带一份可直接喂 DUE 的例子（`dt=0.001` s，物理时长 0.999 s）：

```
data/example/due_N100_T1000.npz
data/example/due_N100_T1000_matrices.png   # Y、U 的 (N,T) 热图
data/example/due_N100_T1000_samples.png    # 若干样本的 Y 曲线
```

重新生成：

```bash
python -m maglev_sim --n-samples 100 --n-times 1000 --dt 0.001 --dtype float32 --r-std 2e-4 --plot --out data/example/due_N100_T1000.npz
```

## 放大到 1000 条 × 10000 点

同样格式，一条命令即可：

```bash
python -m maglev_sim --n-samples 1000 --n-times 10000 --dt 0.001 --dtype float32 --dry-run
python -m maglev_sim --n-samples 1000 --n-times 10000 --dt 0.001 --dtype float32 --out data/due_N1000_T10000.npz
```

| 项目 | 100×1000（示例） | 1000×10000 |
|---|---|---|
| `Y` | `(100, 1000, 3)` | `(1000, 10000, 3)` |
| `U` | `(100, 1000, 2)` | `(1000, 10000, 2)` |
| float32 体积 | 约 0.9 MB（压缩后） | 约 0.20 GB |
| 物理时长 | 0.999 s | 9.999 s |
| 墙钟（约） | 约 1 分钟 | 约 2 小时 |

通道变少不会加快积分：每条样本仍是一次 `solve_ivp`。测试套件不会跑 1000×10000。

## 注意

1. 样本靠 `--z-m-std`、`--r-std` 区分初值和平顺；`--r-std` 给每条样本一个常值 \(r\)，所以 `U[:,:,0]` 各行不同。外力默认为 0，需要受迫振动时改 `configs/demo.yaml` 的 `force`。
2. 间隙越限的轨迹会丢弃重试。
3. `split` 不在一条轨迹中间切开。
4. 演示参数不能当作高速磁浮结论。

## 配置

[configs/demo.yaml](configs/demo.yaml) 只服务积分器。改梁是否启用、外力、是否移动悬浮点即可。
