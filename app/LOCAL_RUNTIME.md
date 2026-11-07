# V1.0.2 本机独立运行空间

2026-11-07 已将当前服务使用的代码与环境收进本目录。删除其他实验版本，不再影响本版的 Python 包、模型权重、原生计算扩展或网页资源。算法、权重、科学参数和既有原生二进制均未因环境迁移而改变；没有创建新版本。

## 必须保留

保留整个 `release/V1.0.2/`，复制或备份时包含隐藏目录：

| 位置 | 用途 |
| --- | --- |
| `.conda/env/` | 独立复制的科学 Python3.10 及全部现有依赖，约8.7GiB |
| `.venv-web/` | 网页 Python3.12 可执行文件与网页依赖，约54MiB |
| `runtime/`、`native/`、顶层 `.py` / `.so` | 推断代码、权重、原生源码与已验证扩展 |
| `web/` | 当前服务使用的网页与 HTTP 后端 |
| `technical-report/` | 同源技术报告和图表、下载数据 |
| `scripts/` | 独立启动入口与环境检查 |
| `.runtime/` | 本版缓存、服务日志与 PID；缓存可以重建，删除会增加首次运行开销 |
| `runs/web/` | 历史和新提交的任务输入、结果、日志；需要保留下载记录就保留它 |
| `evidence/`、`tests/`、各说明和清单 | 验证、来源及完整性记录 |

原 `output/web-jobs-v101/` 的76条记录已迁入 `runs/web/`，任务ID与结果链接保持有效。输入数据本身仍由调用者提供，项目 `data/` 及其链接目标没有修改。历史报告中的来源路径、编译调试字符串及旧实验记录是溯源信息，不是当前运行时依赖。

## 启动

```bash
cd /home/student_su/publish_code/research-workbench/subprojects-single/scalable-neural-phylogeny/release/V1.0.2

# 环境、权重和加载路径检查
bash scripts/run_local.sh doctor --gpu 0

# 推断示例；输出目录必须尚不存在
bash scripts/run_local.sh --task gene --input /absolute/alignment.phy --out runs/gene-001
bash scripts/run_local.sh --task species --input /absolute/alignment.phy --out runs/species-001

# 网页服务：http://127.0.0.1:8765
bash scripts/start_web.sh --port 8765

# 本机调试时允许从 VS Code 侧边栏拖入服务器文件
bash scripts/start_web.sh --port 8765 --allow-server-files
```

不要同时在同一端口启动第二个服务。当前服务的 PID 写在 `.runtime/web-server.pid`，日志在 `.runtime/web-server.log`。网页仍为单 worker、GPU0、CPU16、NUMA0；任务默认位于 `runs/web/`。静态报告由 `/report/` 提供。

科学启动脚本使用本版 `.conda/env/bin/python -I`，清除外部 Python 和共享库路径覆盖，将 Triton、Torch、CUDA、临时目录等缓存固定在 `.runtime/cache/`。网页启动脚本使用 `.venv-web/bin/python -I`，推断子进程再调用同一科学启动脚本。无需激活其他 Conda 环境，也无需联网下载权重或编译扩展。

## 环境来源与边界

科学环境从原服务实际使用的 `release/v-0.1.3/.conda/env/` 离线复制，Conda使用独立文件复制；随后保留源环境实际安装的 `site-packages` 内容，避免历史 Conda 元数据覆盖实际的 pip 库版本。网页依赖从原 `output/web-runtime-v101/env/` 复制。本版没有软链接到这些目录，也没有运行时借用它们的 Python 包或共享库。

实测科学版本：Python3.10.18、Torch2.5.1+cu121、Triton3.1.0、NumPy2.1.2、SciPy1.15.3、ete3 3.1.3。源环境含重复或过时的distribution元数据，旧 `environment-tested.json` 的 Triton/NumPy 快照和实际导入版本不同；本次以迁移前后实际导入为准，保留原有文件，未升级库。各环境元数据清单见 `environment-scientific-packages.json`、`environment-web-packages.json`；科学清单的 `actual_imports` 单独记录核心库实际版本，清单不是可直接用于重装的锁文件。

本次保证的是**当前服务器、当前绝对路径下可独立于其他实验目录运行**。仍使用系统 Linux/glibc、NVIDIA 驱动和 `numactl`；网页环境的 Python3.12 标准库来自系统 `/usr/lib/python3.12`。这些系统组件不在其他实验目录内。未承诺任意移动目录或换主机即可运行；Conda前缀、虚拟环境启动脚本和原生 ABI 仍有路径及硬件约束。

## 迁移验证

- 6组完整迁移前后对照：基因树默认模型、JC、支持度关闭，物种树单矩阵及多locus；全部最终及阶段 Newick 逐字节相同，支持度数值相同。支持度 JSON 仅有输出路径和计时字段差异。
- 52项科学单元测试、AP-NJ独立脚本的7项测试、设计检查40项、并列匹配检查，以及26项网页测试通过。
- 全流程文件访问跟踪与加载检查见 `evidence/STANDALONE_RUNTIME.json`；历史73个成功任务逐一验证下载，新基因树和多locus物种树任务通过真实 HTTP 提交、运行与下载。

详细迁移工作证据位于 `.standalone-migration/20261107/`。交付文件的 `MANIFEST.json` 排除本机环境、缓存、任务和迁移工作目录；环境独立性由 `doctor` 与迁移证据单独检查。
