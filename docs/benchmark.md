# 固定悬浮点时变输入 benchmark：文件职责与使用说明

## 与原项目、现有数据和 DUE 的关系

benchmark 是原有单点磁浮–梁耦合仿真器上的标准化数据生成流程，负责组织工况、
检查数值收敛、生成和划分轨迹，以及记录可复现信息。这里的 benchmark 指数据集
基准方案，不是运行速度测试。它沿用 `plant/` 的物理模型与演示参数。

本地已有的 `data/due_v2_benchmark/` 使用这套流程生成；
`data/due_large_10000/` 使用扩充脚本在前者的基础上生成并合并独立轨迹。
原来的 `data/example/` 示例和 `generate_due_dataset` 入口仍然保留。
数据目录名称是脚本默认值；目录名本身不能证明生成来源，应结合相邻 manifest
和生成合同核对。生成数据默认被 Git 忽略，不随本仓库源码提交。

**生成这些数据时使用 benchmark 代码；把成品 NPZ 交给 DUE 训练时，不需要运行
benchmark、扩充脚本或测试文件。** DUE 按数据格式读取 NPZ 即可；若使用本项目
的读取函数，则需要 `maglev_sim.due.load_due_dataset`，但仍不需要调用 benchmark。

## 各文件的作用

下表中的“参与生成”指被生成命令调用，或作为其底层依赖。

| 文件 | 作用 | 参与生成 | DUE 读取成品时需要 |
|---|---|---|---|
| `docs/benchmark.md` | 说明文件职责、生成方法、数据格式和使用边界 | 否 | 否，供使用者查阅 |
| `maglev_sim/benchmark.py` | 核心生成器：定义工况和随机流、合成时变输入、执行收敛检查、调用原仿真器、筛选轨迹、分配数据集、导出 NPZ/manifest，并管理断点续跑 | 是 | 否 |
| `scripts/generate_benchmark.py` | 命令行薄封装，设置包搜索路径并调用 `benchmark.main`，支持从任意工作目录启动 | 是，使用此脚本启动时 | 否 |
| `scripts/extend_benchmark.py` | 校验基础数据、源码及收敛合同，以独立种子生成更多 ID 轨迹，去重合并，保留原有 OOD 数据，并记录扩充进度与合同 | 是，扩充数据时 | 否 |
| `maglev_sim/due.py` | 原有 DUE 数据生成和读取接口；benchmark 复用其中的 `Y_NAMES`、`U_NAMES`，保持通道定义一致 | 是，提供通道定义；benchmark 不调用其中的旧生成函数 | 仅使用 `load_due_dataset` 时需要 |
| `maglev_sim/plant/config.py` | 底层配置与演示参数；新增不平顺时间节点 `t`，供时间序列输入使用 | 是 | 否 |
| `maglev_sim/plant/excitation.py` | 构建不平顺和外力，进行时间插值，并检查输入数组、时间节点及积分区间覆盖 | 是 | 否 |
| `maglev_sim/plant/model.py` | 原有耦合方程、观测与初始状态；支持按初始时刻的不平顺计算平衡状态，并在积分时使用当前时刻输入 | 是 | 否 |
| `maglev_sim/plant/simulate.py` | 原有闭环 ODE 积分；积分前校验输入覆盖范围，返回轨迹及完成状态等信息 | 是 | 否 |
| `tests/test_benchmark.py` | 验证分层划分、随机输入可复现性、收敛比较规则、导出格式、输入一致性和续跑等行为 | 否，仅测试 | 否 |
| `tests/test_extend_benchmark.py` | 验证扩充批次合并、划分保留、样本 ID 映射、重复内容拒绝和合同校验等行为 | 否，仅测试 | 否 |
| `tests/test_temporal_input.py` | 验证时变输入插值、初始平衡、非法输入拒绝，以及原有空间不平顺行为 | 否，仅测试 | 否 |
| `tests/test_beam.py` | 原有梁模型基础测试：单模态振动和静力挠度收敛，为底层模型提供验证；不同于 benchmark 的逐工况收敛检查 | 否，仅测试 | 否 |

## 数据生成调用关系

```text
scripts/generate_benchmark.py 或 python -m maglev_sim.benchmark
  → benchmark.main
  → ensure_convergence：比较积分精度、梁模态数，确定统一生产配置
  → generate_stage：工况分配、初值和输入抽样、轨迹生成、导出
      → Excitation.from_config → initial_state → simulate
      → NPZ（DUE 数据）+ manifest（来源和质量记录）

scripts/extend_benchmark.py
  → 校验基础 ID/OOD 数据、收敛报告与当前源码
  → 复用基础 ID，调用 benchmark.generate_stage 生成后续独立批次
  → 去重合并 ID，复制已锁定的 OOD 数据
```

测试文件不在上述生产调用链中。它们验证实现行为，不替代正式生成前的收敛检查。

## 从本仓库根目录生成和扩充

```powershell
# 先检查收敛并生成 32 条试运行数据
python scripts/generate_benchmark.py --stage pilot --out-dir data/due_v2_benchmark --workers 4

# 生成 1000 条 ID 数据，以及三类 OOD 数据
python scripts/generate_benchmark.py --stage id --out-dir data/due_v2_benchmark --workers 4
python scripts/generate_benchmark.py --stage ood --out-dir data/due_v2_benchmark --workers 4

# 复用基础 ID 的 1000 条，再生成 9 批各 1000 条，合并为 10000 条
python scripts/extend_benchmark.py --base-dir data/due_v2_benchmark --out-dir data/due_large_10000 --batches 10 --workers 8
```

扩充脚本要求基础目录已有通过的 `convergence.json`、`id.npz`、三类 OOD NPZ
及各自 manifest。基础目录和扩充目录必须独立且不能相互嵌套。
默认 `--base-seed=2026092300`，第一批使用基础 ID 的原种子，后续批次使用
`base_seed + i`（`i=1,...,batches-1`）。默认总划分为训练 7000 条、验证
1500 条、测试 1500 条；OOD 直接保留基础数据，不随 ID 扩充。

已有输出会校验源码版本、内容哈希和生成合同。尤其是 Git 提交会改变记录中的
commit，即使 Python 文件内容相同，也可能使旧目录无法直接续跑。
旧数据仍可交给 DUE 使用；重新生成时应使用新的输出目录。扩充还要求基础数据
与当前源码来源一致，因此不能简单换一个扩充目录来绕过基础数据的版本不匹配。

## 输出文件分别做什么

| 输出 | 用途 | 交给 DUE 是否必需 |
|---|---|---|
| `id.npz` | 分布内训练、验证和测试轨迹，使用 `split` 选择整条轨迹 | 做常规训练/评估时需要 |
| `pilot.npz` | 32 条小数据，用于检查加载、通道和训练流程 | 否，试运行可用 |
| `ood_amplitude.npz` | 输入幅值超出训练分布的泛化测试 | 做该项评估时需要 |
| `ood_frequency.npz` | 输入频率超出训练分布的泛化测试 | 做该项评估时需要 |
| `ood_long.npz` | 更长时间轨迹的泛化测试 | 做该项评估时需要 |
| `*.manifest.json` | 按 NPZ 行序记录样本初值、输入系数、种子、质量、配置及哈希等来源信息 | 读取张量不需要，建议随数据保留 |
| `convergence.json` | 收敛结果、误差和统一生产配置；生成/扩充流程用它校验前提 | 否，建议归档 |
| `.shards/`、`.convergence/` | 基础生成的逐轨迹与收敛断点缓存 | 否，续跑时使用 |
| `.batches/` | 扩充过程的独立批次及其断点数据 | 否，扩充续跑时使用 |
| `large_generation_contract.json` | 锁定扩充的源码、脚本哈希、基础数据、种子和配置 | 否，扩充续跑时使用 |
| `large_generation_status.json` | 扩充阶段、已完成数量和完成/失败状态 | 否，监控时使用 |

本地目录还可能包含 PNG、`DATASET_REPORT.md` 或 `audit.json` 等辅助文件；上述
两个生成入口不自动生成这些文件。它们用于人工查看或额外审计，不是 DUE 读取依赖。

## 将数据交给 DUE

下面以扩充后的 ID 数据为例；DUE 侧也可以直接用 NumPy 加载相同 NPZ 格式。

```python
from maglev_sim import load_due_dataset

d = load_due_dataset("data/due_large_10000/id.npz")
train = d["split"] == "train"
Y_train, U_train = d["Y"][train], d["U"][train]
```

`Y` 的三个通道为车体位移、电磁铁位移、悬浮点梁位移；`U` 的两个通道为
不平顺和梁上外力。用历史段 `Y` 和已知 `U` 预测未来段 `Y`；不要将未来梁位移
作为已知输入。划分应按 `split` 选择整条轨迹，再在各子集内切时间窗口。
benchmark 不包含 DUE 网络训练代码。

## 使用边界

这套流程适合组织可复现的动力学预测实验和泛化评估。数值收敛用于确认积分精度
和模态截断的影响足够小，不能代替真实车辆物理参数与试验验证。当前仍是演示
参数、固定悬浮点工况，不能直接作为已验证的高速磁浮车辆或沿线路移动场景。

## 数值与数据合同细节

The benchmark uses the unchanged demo physical parameters, fixed magnet position
9.12 m and force position 15.12 m. These demo parameters do not constitute a
validated high-speed maglev vehicle model. The existing example and legacy
`generate_due_dataset` API remain available.

From the parent Code directory:

```powershell
python MaglevCoupledDynamics/scripts/generate_benchmark.py --stage convergence --out-dir MaglevCoupledDynamics/data/due_v2_benchmark --workers 4
python MaglevCoupledDynamics/scripts/generate_benchmark.py --stage pilot --out-dir MaglevCoupledDynamics/data/due_v2_benchmark --workers 4
python MaglevCoupledDynamics/scripts/generate_benchmark.py --stage id --out-dir MaglevCoupledDynamics/data/due_v2_benchmark --workers 4
python MaglevCoupledDynamics/scripts/generate_benchmark.py --stage ood --out-dir MaglevCoupledDynamics/data/due_v2_benchmark --workers 4
```

Inside this repository, `python -m maglev_sim.benchmark` accepts the same options.
The default master seed is 20260922. `--stage all` runs convergence and all five
dataset exports. `--max-attempts` bounds simulation rejection retries (default
20). Workers are independent processes; one is the default.

Convergence is mandatory, including for the pilot. Twelve seeded cases compare
RK45 `(rtol=1e-8, atol=1e-10, max_step=.001)` with
`(1e-9, 1e-11, .0005)`, and five with seven beam modes. If needed, nine modes
are checked for every case. The shared production choice is the tighter
integrator and the highest converged modal order. Added modes start at their
static equilibrium; vehicle initial deltas and both complete inputs are
identical between resolutions. Comparisons cover the three outputs and gap
reconstructed from outputs and input. They require finite complete runs,
matching saturation status (fraction tolerance .001), maximum error at most
`max(1e-6 m, .01 * dynamic peak-to-peak)` and RMS error at most
`max(2e-7 m, .005 * dynamic RMS)`. The report includes all tolerances and errors.
Failure through nine modes stops generation. Optional
`--input-grid-diagnostics` evaluates the same continuous waveform coefficients
on .0005 s input knots, retaining the .001 s observation grid. Use this option
consistently on every invocation sharing an output directory.

Temporal irregularity and point force use strictly increasing finite time
knots covering the full integration interval, with linear interpolation. The
benchmark saves exactly those input knots on the observation grid. No
irregularity derivative term is added to the force or PID equations.

Each active input is a sum of eight independent uniformly drawn frequencies
and phases, multiplied by a .2 s half-cosine ramp. Its RMS after the ramp is
normalized to the drawn amplitude; crest factors exceeding four are rejected
and logged. Streams for initial conditions, irregularity, force, membership
shuffle and attempts are independent deterministic SeedSequence namespaces.
Initial displacements have sigma 1e-4 m, velocities 1e-3 m/s, each Gaussian
rejected outside three sigma. Other states remain at equilibrium.

Exports are `pilot.npz` (32), `id.npz` (1000), `ood_amplitude.npz` (100),
`ood_frequency.npz` (100), `ood_long.npz` (50). Standard runs have 5001 samples;
long runs have 10001. ID splits are 700/150/150, each exactly stratified into
10/20/20/50 percent free/r-only/P-only/both. Pilot uses 22/5/5 nonempty splits.
OOD is entirely test data. Amplitude and frequency OOD have both inputs active
and change just one input factor, labelled in each sample's `group` field.

NPZ files contain exactly `Y`, `U`, `t`, `dt`, `split`, `Y_names`, `U_names`.
Y and U are float32; time is float64; split is Unicode and requires no pickle.
The adjacent `.manifest.json` contains `samples` in exact NPZ row order, with
`id`, `category`, `group`, `split`, initial deltas, waveform coefficients,
subseed provenance, attempts, quality statistics and hashes. Top-level config,
source commit and source-content hashes, convergence signature, aggregate
quality and runtimes provide reproducibility. Internally, `.shards` and
`.convergence` retain atomic per-run checkpoints. Resuming validates source,
config, seed, manifest and payload hashes. Incompatible output directories
raise an error; use a new directory for changed contracts.
