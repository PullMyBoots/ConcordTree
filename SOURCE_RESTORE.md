# 从工作站保存的发行快照恢复

工作站目录：`D:\serve-backup\ConcordTree\distribution-test`。

完整运行环境保存在 `concordtree-1.0.2-linux-x86_64-cuda121.tar.gz`；发行源码、模型及 Git 历史保存在 `concordtree-1.0.2-linux-r1-source.bundle`。两者各有 `.sha256` 校验文件。完整研究工作台的灾备归档另行管理。

## 恢复并启动软件

在 Linux 或工作站 WSL 终端执行。先进入备份目录，再校验并解压到 Linux 文件系统中的新目录：

```bash
cd /mnt/d/serve-backup/ConcordTree/distribution-test
sha256sum -c concordtree-1.0.2-linux-x86_64-cuda121.tar.gz.sha256 && \
mkdir -p "$HOME/ConcordTree-restored-1.0.2" && \
tar -xzf concordtree-1.0.2-linux-x86_64-cuda121.tar.gz -C "$HOME/ConcordTree-restored-1.0.2" && \
"$HOME/ConcordTree-restored-1.0.2/concordtree-1.0.2-linux-x86_64-cuda121/concordtree"
```

打开 `http://127.0.0.1:8765`。需要其他端口或 GPU 时，在最后的启动命令后添加 `--port 8876 --gpu 1`。运行要求和已验证设备范围见 README.md；稳定版默认 Web，包含科学依赖和模型。

## 恢复发行源码与版本历史

```bash
sha256sum -c concordtree-1.0.2-linux-r1-source.bundle.sha256 && \
git clone --branch distribution/v1.0.2-linux-r1 \
  concordtree-1.0.2-linux-r1-source.bundle "$HOME/ConcordTree-source-restored"
```

源码快照记录算法、模型、Web 静态资源、分发脚本和验证报告。逐次复测记录在 `regression-trials.json`；现代运行栈的独立诊断在 `MODERN_VALIDATION.md` 和 `modern-regression-results.json`。

在线安装器的 `--from DIRECTORY` 模式使用五个发行分片及对应校验清单。此工作站保存的是完整 portable 归档，按上面的解压步骤恢复。
