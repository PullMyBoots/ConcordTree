# ConcordTree Web

极简双页界面，仅提供基因树和序列到物种树两种任务。
全英文提交页包含 Gene Tree / Species Tree、必要选项和 Upload，无额外标题或常驻预览提示。
选择或拖入合法文件后自动上传并提交，无 Infer 按钮。服务端确认接收后才显示 Queued。
上传卡片内有默认折叠的队列状态栏，点击或用键盘展开 Running / Queued 任务格子；
折叠时保留服务器状态和任务摘要，结果页内也保留同一入口与展开状态。
运行格子柔和闪动，等待格子依次填充；系统减少动态效果时关闭动画。
每 5 秒更新，隐藏标签页暂停刷新。`GET /api/queue` 仅返回汇总数字和接单状态，
不返回他人的任务标识或文件名。未启用推断显示 Server paused；掉线显示
Server offline、断开图标和未知计数，不把未知状态当作空队列，恢复连接后自动更新。
提交前检查服务器状态，暂停、离线或队列满时不上传。上传中锁定控件以避免重复提交；
上传响应丢失时显示未能确认，不自动重传，以免重复排队。
零文件拖放与多文件拖放分别提示；默认仅接受真实浏览器文件，临时调试模式额外接受 VS Code 服务器文件路径。
Upload 下方的 Try example 可直接加载内置小型样例并按当前参数提交，无需准备本地文件。
Queued 页面显示当前等待队列位次，随任务完成或取消更新；不估计未经测量的等待时间。
基因树可选择进化模型和分支支持度，物种树仅提供分支支持度开关。
结果页显示任务状态、实际推断阶段、可滚动、缩放和拖动的在线树预览、
Newick 下载、返回提交页。Back 返回新的 Upload，保留任务类型。
结果地址使用 `#job=<id>`，刷新后读取已有任务。超过 400 个物种的树保留完整下载，
在线预览显示数量限制提示。无任务记录、示例树、基因树合并或前端构建依赖。

默认 Geometry 主题由 Canvas 实时绘制不对称的曲线生命树，逐级生长、分叉，
枝端浮现原创线描动物与植物图标；手机版将四个图标安排在上传区域上方。
每轮约 23 秒，完成后淡出重生。适配高像素密度屏幕，左下角提供暂停和主题切换。
Nature 主题每次进入随机选一张本地摄影壁纸，不自动轮播，只加载选中的照片。
左下角换图按钮手动选择下一张，避免立即重复；减少动态效果时仍可换图。
图库 `backgrounds.json` 包含 12 张特写：变色龙、虎、蜂鸟、恐龙复原、菊石、
蕨叶、茅膏菜、蘑菇、古植物标本、硅藻、植物细胞和病毒显微图像。
森林和珊瑚景观已退出随机图库。大多数下载原图长边为 4096 像素；
古植物标本为 3000 × 2252，病毒电镜为 2097 × 2097，并非所有素材均为原生 4K。
桌面表单固定在右侧，不随换图移动；恐龙图片水平镜像，使主体留在左侧。
桌面与手机分别提供 WebP 构图，主体置于表单旁或上方，周围使用同图模糊延展，
不对主体做 AI 放大或生成。每次只请求所选设备对应的图片版本。
右下角提供小型署名和授权链接，完整来源及媒体类型见 `/credits.html`；
恐龙复原和显微图像均明确标注，古植物馆藏项目的模型分类也保留说明。
主题选择保存在浏览器本地；遵循系统减少动态效果设置，隐藏标签页暂停绘制。
摄影来源及授权见 `assets/CREDITS.md`，素材尺寸与校验值见 `assets/SOURCES.json`。
摄影素材仍可在用户生成新壁纸后替换，不含外部字体或脚本依赖。

## 运行

当前规范运行副本为 `release/V1.0.2/web/`。从本版目录使用本地环境启动：

```bash
cd /home/student_su/publish_code/research-workbench/subprojects-single/scalable-neural-phylogeny/release/V1.0.2
bash scripts/start_web.sh --port 8765
# 打开 http://127.0.0.1:8765
```

网页依赖已经独立放入本版 `.venv-web/`；科学依赖在 `.conda/env/`，无需其他发布或实验目录。清理与系统依赖边界见 [LOCAL_RUNTIME.md](../LOCAL_RUNTIME.md)。

默认接收真实任务；`--preview` 只显示网页并暂停接单。
接口通过本版 `scripts/run_local.sh` 调用 `run.py`，使用本版 `.conda/env/bin/python`；单 worker、GPU0、CPU16、NUMA0。
输入、输出和执行日志位于本版 `runs/web/<job-id>/`，已有任务已迁入；可用 `--runs` 指定项目内其他非 data 输出目录。
科学缓存与网页上传临时文件均在本版 `.runtime/cache/`。
上传上限 100 MiB；补齐后的矩阵上限为 100 MiB 个字符；最多 16 个运行或排队任务，
每个任务 15 分钟超时。上传解析在线程中执行，避免阻塞队列查询。
实际任务失败会展示失败，不返回伪造结果。
服务退出会停止其计算子进程；重启不自动重跑未完成任务。枝长和物种树的科学边界沿用发布版文档。

## 输入与选项

- **Gene Tree**：单个 PHYLIP 对齐矩阵，至少 4 个 taxa；拒绝多 locus 输入。
  模型默认为 `GTR+FO+I+G4`，另可选择 `GTR+I+G4`、`GTR+G4`、`HKY+G4`、
  `GTR`、`SYM`、`TN93`、`HKY`、`K80`、`F81`、`JC`。
- **Species Tree**：一个 PHYLIP/PAML 文本文件，每个 locus 使用独立的
  `序列数 位点数` 头和对应 MSA；总计至少 24 个 taxa。
  一个块自动用 `msa_only`；两个及以上块自动生成边界并用 `observed_loci`。
  不从已拼接的单矩阵猜测隐藏的基因边界，不提供手动分区选项。
- **Branch support**：两类任务默认勾选；取消会向后端传入 `--no-support`。
  勾选时网页在内枝旁显示数值，下载的 Newick 也保留支持度。
  基因树为 Local likelihood weight，物种树为 Quartet margin index；
  页面显示的 0–100 是支持度指标，不是经过校准的分支正确概率。

接受 relaxed sequential（包括序列换行和 PAML 的名称独占一行）及 interleaved PHYLIP；
序列名称与序列之间须有空白，不支持无分隔符的传统固定十字符名称行。
接受 UTF-8/BOM、空行、DNA IUPAC 字符及 `N ? - .`；不接受 FASTA、蛋白质或未对齐序列。
每个 locus 内名称必须唯一，不同 locus 通过相同名称对应物种；缺失物种在该 locus 补 `?`。
每个物种在全部输入中至少须有一个 A/C/G/T。文件扩展名仅用于文件选择器提示，服务端按内容验证。

多 locus 形式示意（此处仅为格式演示，实际物种树至少需要 24 个 taxa）：

```text
2 8
species_a ACGTACGT
species_b ACGTTCGT

2 6
species_a AAGGCC
species_b AAGGCT
```

## 验证

```bash
.venv-web/bin/python -B -m unittest web.test_server -v
node --check web/app.js
```

当前26项输入/API测试在队列调度边界mock，不运行科学子进程。独立环境迁移及真实HTTP回归见 [STANDALONE_RUNTIME.json](../evidence/STANDALONE_RUNTIME.json)。
另已通过真实 HTTP 和浏览器提交任务，在 GPU0 验证基因树模型、支持度开关、
物种树自动分区、预览、下载、刷新恢复及移动端布局。
证据见 `VALIDATION.json` 和 `output/web-v101-integration-20261105/REPORT.md`；
浏览器检查与截图放在 `output/playwright/`。小型集成用例不代表大数据性能测试。

## 边界

当前服务绑定 `127.0.0.1:8765`，用于本地开发；公网部署尚未完成。
任务文件保留在本机，尚未提供账户、用户隔离或自动清理。
本版 `runs/`、`.runtime/`、`.conda/` 和 `.venv-web/` 不作为网页静态资源公开。


## VS Code 远程侧边栏拖入（临时调试）

```bash
bash scripts/start_web.sh --port 8765 --allow-server-files
```

刷新页面后，可直接拖入本项目的服务器 PHYLIP/PAML 文件。兼容 VS Code 的
ResourceURLs、CodeFiles、CodeEditors、URI list 及绝对路径文本；一次仅接受一个文件。
浏览器只传路径，后端只读原文件并在任务目录保存规范化副本，不经本地电脑传输 MSA。
仅允许项目及 data 中已登记的软链接输入目标，隐藏文件和项目外其他路径被拒绝。
只读取磁盘上已保存内容，不读取编辑器中未保存的改动。
服务器文件及补齐矩阵上限为 512 MiB；普通上传仍是 100 MiB，任务超时仍为 15 分钟。
不带 --allow-server-files 启动时路径接口返回 403。该开关用于本地调试。

截图中的 mammal241 MSA.phy（112368213 字节、241 taxa × 466232 位点）
已通过只读输入适配与格式校验，未为此启动大文件推断。
两类任务通过模拟 VS Code 原生拖放格式的真实后端测试；实际 VS Code 内嵌浏览器的
系统拖放仍由用户刷新页面后检验。


## 树预览布局

叶节点使用深度优先顺序，保证各子树叶片连续；无根二叉树的最外层三支不是多分叉。
画布根据叶数与最长物种名调整尺寸；大型图在卡片内滚动，普通滚轮不再意外缩放。
Ctrl/Command + 滚轮缩放，Reset View 恢复初始缩放与滚动位置。
支持度标签避让，并用细虚线指向发生位置调整的分支。
该布局修复只影响显示，保留下载的拓扑、枝长与支持度原始值。


## 完成页用时

推断成功后在结果摘要下显示 `Inference time`，取后端保存的
`elapsed_seconds`（算法 `RESULT.json.file_to_tree_seconds`），不含上传、排队和进程启动。
短任务显示秒数，较长任务自动显示分钟或小时；刷新后仍可查看。
浏览器验证复用三个已完成的真实任务，不新增推断，检查失败任务隐藏、刷新保留和移动端布局。
证据见 `output/gene-long-alignment-fix-20261105/BROWSER_TIMING.json`。


## 按需预览与推断进度

完成页默认收起树预览和支持度说明。taxa ≤400 时显示 Preview，点击后才绘图；
taxa >400 时隐藏预览按钮和画布，保留完整树下载与推断用时。
推断期间显示阶段进度条，条内呈现后端最新阶段、轮次及分支调整数；
Activity 可展开查看本次页面实际收到的最近50条不同记录，刷新后重新收集。
阶段刻度不代表时间百分比，不估算剩余时长；排队、断线与失败有对应状态。
验证见 `output/web-preview-progress-20261106/BROWSER.json`：真实已完成结果复用，
运行中状态及400/401边界通过浏览器拦截API响应验证，未新增GPU推断。


## 本地技术报告

Technical Report 指向同源 `/report/`，由 `release/V1.0.2/technical-report/`
提供四个独立页面：`index.html`（文件格式）、`accuracy.html`（准确性）、
`performance.html`（速度与消耗）和 `method.html`（方法）。顶部导航切换页面，
默认英文，可切换中文且跨页保留语言；阅读、图表及语言切换不请求外网。
基因树 PHYLIP 和物种树 PHYLIP/PAML 示例可在格式页下载。

布局按用户的 `report.TXT` 调整。已移除论文来源的 A10/A20/A50 与 RNASim
数据准确率、耗时和对应说明，DIPPER 软件在项目其他数据上的对照仍保留。
历史准确性及规模测试标明适用配置；补充枝长与支持度既有验证结果，
方法页描述当前共享 AP-NJ / MLP-NNI 与分别精修的流程，内存测量缺失明确标注。
未新增推断实验，未创建发布版本，也未修改算法实现。

`SNAPSHOT.json` 区分原始下载来源与本地修改后的文件，原报告备份在
`output/report-layout-20261106/before/`，不通过报告路由公开。
25项API测试通过；四页、中英切换、桌面/390/320宽度、样例格式及剔除内容
的验证见 `output/report-layout-20261106/`；截图在 `output/playwright/`。
