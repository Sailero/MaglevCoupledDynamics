# DUE 示例（100 × 1000）

`due_N100_T1000.npz` 可直接给 DUE 用，键只有：`Y` `(100,1000,3)`、`U` `(100,1000,2)`、`t`、`dt`、`split`、`Y_names`、`U_names`。

```python
from maglev_sim import load_due_dataset
d = load_due_dataset("data/example/due_N100_T1000.npz")
Y, U = d["Y"], d["U"]
```

同目录两张图：`*_matrices.png` 是矩阵热图，`*_samples.png` 是样本曲线。
