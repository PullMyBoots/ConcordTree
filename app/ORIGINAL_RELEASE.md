# ConcordTree V1.0.2

本版已在当前服务器完成独立运行环境迁移：科学环境、网页环境、网页代码、权重、原生扩展、缓存和任务均位于本目录。启动与清理边界见 [LOCAL_RUNTIME.md](LOCAL_RUNTIME.md)，验证记录见 [STANDALONE_RUNTIME.json](evidence/STANDALONE_RUNTIME.json)。保留整个目录，包括隐藏的 `.conda/`、`.venv-web/` 和 `.runtime/`。

统一观测分母、精确并列最优匹配与初始化证据修复见 [APNJ_DESIGN_REPAIR.md](evidence/APNJ_DESIGN_REPAIR.md)。初始树准确率提升目标未达成；常规初始化有小幅开销，最终开发集均值与逐记录损益均完整保留。

长序列内核修复与数值差异边界见 [PATCH_NOTES.md](PATCH_NOTES.md)。
现有 V1.0.2 的工程健壮性补强、尺寸边界回归及速度对照见
[KERNEL_ROBUSTNESS.json](evidence/KERNEL_ROBUSTNESS.json)；这些历史内核修复保持当时的算法约定与权重。

当前初始化采用**全位点有界 AP-NJ**：稀疏 Q 候选合并、固定512子树的标准 NJ 收尾、四轮固定距离 NNI 整理。初始阶段使用全部原始位点与共同观测距离；基因树使用 K2P-gamma，物种树使用 JC。MLP、后段与支持度保持原实现；输入输出和版本号不变。本轮准确率、速度及其边界见 [APNJ_INITIALIZATION.md](evidence/APNJ_INITIALIZATION.md)，复杂度证明见 [APNJ_COMPLEXITY.md](evidence/APNJ_COMPLEXITY.md)。

统一的序列到基因树／物种树实现。两种任务调用同一个优化前段，按任务固定选择 MLP 权重，再进入各自后段。

| 阶段 | Gene tree | Species tree |
| --- | --- | --- |
| 初始构树 | 共享有界 AP-NJ，全位点 | 共享有界 AP-NJ，全位点 |
| 前置精修 | 共享 MLP-NNI，**homogeneous** 权重 | 共享 MLP-NNI，**heterogeneous** 权重 |
| 后续精修 | v0.5.9-hybrid 的 V11 全位点 DNA 似然 | species-msc-compact-20261105 的紧凑统计精修 |
| 输出 | 无根拓扑及枝长，单位为每位点期望替换数 | 无根拓扑，不估计枝长 |

这里 homogeneous / heterogeneous 指训练得到的 MLP checkpoint；两者网络结构相同。它不是后段似然模型的自动选择。没有 SplitBank 或基因树合并流程。

## 在当前服务器运行

进入本目录运行；输出必须使用新目录，并且不能写入 `data/`、其链接目标或 `.ResearchProject/`：

```bash
cd /home/student_su/publish_code/research-workbench/subprojects-single/scalable-neural-phylogeny/release/V1.0.2

# Gene tree: homogeneous MLP + full-site GTR+FO+I+G4 likelihood
bash scripts/run_local.sh \
  --task gene --input /absolute/path/alignment.phy --out runs/my-gene-job

# Species tree: heterogeneous MLP + MSA-only statistical refinement
bash scripts/run_local.sh \
  --task species --input /absolute/path/supermatrix.fasta --out runs/my-species-job

# Optional observed locus boundaries for species trees
bash scripts/run_local.sh \
  --task species --input /absolute/path/supermatrix.phy \
  --loci /absolute/path/loci.jsonl --out runs/my-partitioned-species-job
```

网页服务使用 `bash scripts/start_web.sh --port 8765`；本机 VS Code 服务器文件拖入调试可加 `--allow-server-files`。默认任务保存在 `runs/web/`，本地报告仍为同源 `/report/`。完整说明见 [web/README.md](web/README.md)。

输入必须是已经比对的 DNA：顺序 PHYLIP（每行一个名称和完整序列），或等长 FASTA（可跨行）。名称唯一且不含空白；支持 IUPAC、`N`、`?`、`-`、`.`，非 A/C/G/T 字符沿用原实现作为未知状态处理。两种格式均可附加 `.zst` 压缩后缀，需服务器具有 `zstd`。解压和格式转换只写到本次输出目录。拒绝行长不符、重复名称、零位点及完全缺失的采样 taxon；MLP 前段要求至少一个变异位点。基因树最少4 taxa；物种树沿用 nearest24 后段，最少24 taxa。

分区文件按比对列的顺序每行一个 JSON 对象，只使用可观测的 `alignment_length`。长度均为正整数，合计必须恰好覆盖全部位点；不从文件名、参考树或模拟真值推断分区。例如257位点：

```jsonl
{"alignment_length": 128}
{"alignment_length": 129}
```

可选控制：

| 参数 | 默认与作用 |
| --- | --- |
| `--gpu 0` | 物理 GPU 编号，一次进程使用一张 GPU |
| `--cpu-threads 16` | CPU 线程数 |
| `--sample-fraction F` | 保留参数兼容；AP-NJ 始终读取全部位点，不再按此比例采样 |
| `--missing-data-model standard` | 保留旧参数兼容；初始距离固定使用共同观测位点，不作自动路由 |
| `--stop-ratio .005` | 后续精修停止比例；前置 MLP 始终固定 `.005` |
| `--model GTR+FO+I+G4` | 仅基因树，支持 JC/F81/K80/HKY/TN93/SYM/GTR 及合法修饰符 |
| `--rounds R` | 仅基因树，后段轮数预算；CLI 与显式 `--stop-ratio` 互斥 |
| `--views V` | 仅基因树，多初始视图；默认1，每个视图都调用同一共享前段 |
| `--no-support` | 关闭默认的最终分支证据汇总，用于调试或性能对照 |

AP-NJ 使用全部原始位点；多视图改变投影、代表元与landmark采样的固定种子。MLP沿用原始常量位点过滤，基因似然与物种统计评分也读取全位点。

## 输出与服务调用

`prediction.nwk` 是带分支支持指标的最终 Newick；`branch-support.json` 给出每条内枝的值、方法与证据量，`prediction.unannotated.nwk` 保留原始拓扑和枝长文件。`RESULT.json` 给出任务、checkpoint、状态、总耗时和支持指标摘要；`progress.json` 是原子更新的最新进度，可供网页轮询。失败时记录 `status: failed` 并抛出错误，不提供占位树。`initial/` 保存 `apnj.nwk`、MLP 树与完整动作历史；`gene/` 保存似然拟合与精修记录，或由 `species.json` 保存物种后段记录。`runtime.json` 保存设备、线程与缓存路径。

默认开启轻量分支证据汇总，无需重复推断或Bootstrap重建：基因树为 **Local likelihood weight**，物种树为 **Quartet margin index（平均最不利分差）**。二者都是明确限定的0–100局部证据指标，**未经校准，不能解释为正确概率、Bootstrap或ASTRAL localPP，也不共用95等阈值**。通常复用最后一轮搜索；必要时只补做最终状态的局部评分或已缓存证据映射。物种树依据AUC优先目标采用已冻结的分差策略；确认组AUC为0.917083→0.922821，差值区间跨零，不保证每个截断比例都改善。原始分差保存在`mean_worst_margin`，旧一致率另存为`sampled_quartet_agreement`；0–100显示值不是百分比。详见 [BRANCH_SUPPORT.md](BRANCH_SUPPORT.md) 和 [采用记录](evidence/SPECIES_MARGIN_ADOPTION.md)。调试／性能对照可用 `--no-support`，API对应 `branch_support=False`。

API 适合每个服务 worker 保留一个引擎，连续处理两种任务，避免反复加载权重：

```python
import sys
sys.path.insert(0, "/absolute/path/to/project/release/V1.0.2")
from engine import UnifiedEngine

engine = UnifiedEngine(gpu=0, cpu_threads=16)
result = engine.infer("/absolute/path/alignment.phy", "output/job-001", task="gene")
result = engine.infer("/absolute/path/supermatrix.phy", "output/job-002", task="species")
```

`progress=callback` 可直接接收进度字典。一次进程中的推断会串行执行；这是算法接口，网站队列和 HTTP 上传应由服务层管理。使用新 Python 进程加载本版，避免与历史发布版的同名模块混用。程序会检测已加载的冲突运行时。

## 验证与边界

2026-11-07 同版本增加搜索健壮性修复，详见 [SEARCH_ROBUSTNESS.md](SEARCH_ROBUSTNESS.md)。
MLP 完整状态循环会提前退出并交给后段；似然搜索的重复拓扑微小收益停滞、参数拟合的外层重复求解均有独立保护。
默认停止比例保持不变。`RESULT.json.quality` 区分后段搜索收敛、参数收敛和保护介入；完成输出不等同于已证明收敛。
常量序列会明确标记缺乏拓扑信息。网页结果保留相关提示和下载功能。

本轮初始树和完整流程结果见上方 APNJ_INITIALIZATION.md。以下旧版 AP-NJ 等价提速与后段回归属于历史快照，不是本轮有界 AP-NJ 的测量；原 DIPPER 接入证据亦保留为历史记录。

详见 [evidence/VALIDATION.md](evidence/VALIDATION.md)。4096 taxa × 2048位点、L40S/GPU0/16线程/NUMA0，前段同配置热态中位耗时：homogeneous `1.903 → 1.080s`，heterogeneous `1.940 → 1.089s`；相对未优化的 v0.5.3 前段分别为 `1.76×`、`1.78×`。每种权重／输入组合一次预热、三次计时，AP-NJ、MLP结果树和完整动作历史完全相同。**这些不是完整推断耗时或准确率评估**；heterogeneous速度用例也使用相同的基因模拟观测，以隔离实现差异。紧凑物种旧目录本身仅含后段，不存在可直接比较的完整初始构树流程。

六组独立进程回归通过：两组基因树完整流程与原 v0.5.9 在改用相同 homogeneous 权重后树、枝长和搜索历史一致；四组物种后段在相同前置树下与原紧凑实现一致。基因旧版默认是 heterogeneous，按照本次要求换为 homogeneous 是科学配置变更，旧版准确率结论不能直接沿用。历史物种前段与新的共享采样／构树约定不同，也不声称完整物种树复现旧流程。

上面的交付回归在分支标注功能加入前完成。新增支持指标的开关对照、最终分支映射检查和额外耗时另见 [evidence/BRANCH_SUPPORT_VALIDATION.md](evidence/BRANCH_SUPPORT_VALIDATION.md)；数值树标签的变化不等于拓扑或枝长变化。

新目录对旧基因后段增加了一个浮点边界修复：只将超出合法 alpha/pinv 上界或 alpha 下界、且距离不超过4 ULP的值归一到原界限，避免 `exp(log(100))` 的舍入错误。没有扩大科学参数范围，越界值仍拒绝。原始发布目录未修改。

两种后段仍是局部搜索；本次集成没有证明全局最优或一般 MSC 一致性，没有进行新权重配置下的泛化准确率确认。

## 环境与文件完整性

当前本地环境：Linux x86_64、CPython3.10.18、NVIDIA L40S、PyTorch2.5.1+cu121、Triton3.1.0、NumPy2.1.2、SciPy1.15.3、ete3 3.1.3。精确版本见 [environment-tested.json](environment-tested.json)。这些是迁移时对原服务实际导入版本的测量；旧文档中的 Triton/NumPy 快照与实际文件不同，本次没有升级库。不支持CPU单独推断。科学 Python 和依赖已复制至 `.conda/env/`，不再借用其他发布目录。

缓存全部位于 `.runtime/cache/`；标准启动脚本隔离外部 Python 搜索路径和共享库覆盖。项目标识存在时继续保护项目的只读输入；标识不存在时输出根退回本目录。系统 Linux、NVIDIA 驱动以及网页 Python3.12 的系统标准库仍需保留；它们不位于待清理的实验目录。此迁移针对当前服务器，移动绝对路径或换主机不在已验证范围。预编译扩展绑定 CPython3.10 / Linux x86_64。[NATIVE.md](NATIVE.md) 说明原生源码及构建边界；迁移未重新编译或改变这些二进制。

```bash
python3 -B verify_release.py
bash scripts/run_local.sh doctor --gpu 0
.conda/env/bin/python -B tests/test_contract.py
```

[METHOD_CONTRACT.json](METHOD_CONTRACT.json) 固定算法约定；[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) 记录导入来源、原始哈希与集成修改；`MANIFEST.json` 校验交付代码与静态资产，不包含本机环境、运行缓存、任务、迁移工作记录和 Python 字节码。环境版本与独立性另由上述环境清单及 doctor 检查记录。
