---
title: PyBCA Guide (v0.1)
nav_order: 20
---

# PyBCA Guide (v0.1)

PyBCA の推奨実行フローは `Engine` API です。`Config` に条件をまとめて `Engine.run()` を呼び出します。

- Engine API の詳細: [Engine API (PyBCA)](engine_api.md)

## 最小例

```python
from PyBCA.api import Config, Engine

cfg = Config(
    cellspace_path="Sample/Cellspace/C-Join.yaml",
    rule_paths=("Sample/rule/base-rule.yaml",),
    steps=20,
    trials=3,
    global_prob=0.5,
    device="cpu",
)

result = Engine(cfg).run()
```

## 旧来 API (BCA_Simulator)

`BCA_Simulator` を直接使う方法も残っていますが、新規コードでは `Engine` を推奨します。
`BCA_Simulator` は `PyBCA.cli_simClass` から利用できます。

```python
from PyBCA.cli_simClass import BCA_Simulator

sim = BCA_Simulator(
    cellspace_path="Sample/Cellspace/C-Join.yaml",
    rule_paths=["Sample/rule/base-rule.yaml"],
    device="cpu",
)

sim.Allocate_torch_Tensors_on_Device()
sim.set_ParallelTrial(3)
sim.run_steps(steps=20, global_prob=0.5)
```
