# ConcordTree

ConcordTree 使用一张 NVIDIA GPU，直接从已经对齐的 DNA 或 RNA 序列构建无根
系统发育树。

方法先从序列比对中构造多个互补的树视图，合并多个视图共同支持的分支，再用
一个每次比较四个分类群的神经网络检查不确定的局部分支。这里可以选择两种模型
架构：

- **轻量模型（`mlp`）** 是速度更快的默认选项。
- **注意力模型（`transformer`）** 使用更大的注意力网络，计算量也更高。

两种选项使用完全相同的建树流程，只替换负责局部判断的神经网络模型。

**[查看完整测试与对比结果](https://pullmyboots.github.io/ConcordTree/)**

## 运行要求

- Linux x86-64
- CPython 3.10
- NVIDIA GPU，驱动兼容 PyTorch 2.5.1 和 CUDA 12.1
- 推荐使用 Conda 创建独立环境

## 安装

```bash
bash scripts/create_environment.sh
source .conda/env/bin/activate
concordtree doctor --device cuda:0
```

如果已有兼容的 Python 3.10 环境：

```bash
python -m pip install -r requirements-lock.txt
python -m pip install dist/concordtree-0.1.2-cp310-cp310-linux_x86_64.whl
concordtree doctor --device cuda:0
```

## 运行

先用软件自带的合成序列检查安装：

```bash
concordtree infer \
  --msa examples/minimal24.phy \
  --work-dir runs/demo \
  --tree-type gene-tree \
  --quartet-model mlp \
  --device cuda:0
```

推断自己的序列：

```bash
concordtree inspect --msa input.phy

concordtree infer \
  --msa input.phy \
  --work-dir runs/my-tree \
  --tree-type species-tree \
  --quartet-model mlp \
  --missing-data-model standard \
  --device cuda:0
```

输入必须是至少包含 24 个分类群的 sequential PHYLIP 核苷酸比对。程序接受
`U`，并将其按 `T` 处理；简并碱基、缺口以及其他非 `ACGTU` 字符均按缺失数据
处理。最终树写入 `WORK_DIR/tree.nwk`。

## 选择树类型

| 参数 | 适用情况 |
| --- | --- |
| `--tree-type gene-tree` | 单个基因座，或者整个比对共享同一棵基因树。 |
| `--tree-type species-tree` | 拼接的多个基因座可能具有不同的基因树；这是默认值。 |

这个选项只负责选择与任务相匹配的训练权重。ConcordTree 仍然读取一份序列比对，
不要求用户事先分别推断每个基因的树。

## 选择缺失数据模型

先运行 `concordtree inspect --msa input.phy`。这个命令只读取序列比对，并报告
相干稀疏指数（CSI）：

| CSI | 当前建议 |
| --- | --- |
| `≤ 0.25` | 使用 `standard`。 |
| `0.25–0.35` | 证据不足；如果重视准确率，建议两种模式都比较。 |
| `≥ 0.35` | 使用 `coverage-aware`。 |

| 参数 | 适用情况 |
| --- | --- |
| `--missing-data-model standard` | 大多数物种共享大部分对齐位点，缺失主要是普通比对 gap；这是默认值。 |
| `--missing-data-model coverage-aware` | 许多物种对只有少量共同观测位点，而且不同物种倾向于缺失相同区域。 |

这个参数只改变每个 View 初始树使用的距离。神经网络精修和多 View 合并流程完全
相同。

## 选择神经网络模型

| 参数 | 适用情况 |
| --- | --- |
| `--quartet-model mlp` | 使用轻量模型；这是速度更快的默认值。 |
| `--quartet-model transformer` | 使用更大的注意力模型。 |

## 控制精修停止条件

三个精修阶段分别提供一个收敛阈值和一个最大轮数：

```text
--view-stop-ratio             --view-max-rounds
--coordinate-stop-ratio       --coordinate-max-rounds
--saturation-stop-ratio       --saturation-max-rounds
```

本轮没有换边、达到已启用的换边比例阈值、或者用完已启用的轮数预算时，该阶段
停止。每个参数都可设置数值；每一对参数中的任意一个也可设置为 `none`。例如，
`--saturation-stop-ratio 0.002 --saturation-max-rounds none` 表示 Saturation
只按阈值停止。三个阶段的默认组合分别是 `0.01/24`、`0.005/4` 和
`0.005/5`。

`--view-count` 可设为 2–8，默认值为 4。`--parallelism 0` 会自动决定 CPU
线程预算。完整参数见 `concordtree infer --help`。

## 其他命令

```bash
# 无需 GPU 或参考树，检查缺失数据结构
concordtree inspect --msa input.phy

# 检查 CUDA、原生后端和模型文件
concordtree doctor --device cuda:0

# 比较两棵树
concordtree validate \
  --msa input.phy \
  --expected expected.nwk \
  --actual runs/my-tree/tree.nwk
```

[English](README.md) · [测试结果与方法](https://pullmyboots.github.io/ConcordTree/) ·
[许可证](LICENSE)
