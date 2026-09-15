# 从这里开始：本地代码与最小 CPU 包

本页仅保留旧的独立 **CPU 核心子包**说明。如果你从服务器 Git 分支取得代码，
完整新实验入口是 RUN_EXPERIMENT.md，不要只停留在本页的演示检查。
仅解压最小 CPU 子包时没有新训练入口；完整执行包需从指定服务器分支获取。
不要直接运行旧日期的训练脚本来开始新实验。

## 先跑一次，不需要模型

需要 Python 3.10+；在包含本文件的目录运行，使用 `python -m` 保证包路径一致。
冒烟流程只依赖 Python 标准库，不需要联网、PyTorch、tokenizer、GPU 或真实数据。

```text
python -B -m scripts.local_cpu_smoke --config configs/local_cpu.json --output tmp/first_cpu_smoke
```

成功时输出 `local_cpu_smoke_pass`。产物在指定新目录：

- `synthetic/`：24 条合成退款案例及各曝光的演示数据。
- `messages_only/`：可被后续工具读取的 messages-only JSONL 格式演示。
- `smoke.json`：配置、产物哈希及检查状态。

这只是“数据生成 → prompt 拼装 → 格式导出 → oracle 自检”和一个合成 ABCD
规则案例的工程检查。oracle 自检不是模型准确率；没有训练，不能作为论文结果。
已有输出目录会被拒绝，请换新目录，不要先删除旧产物。

## 测试

建议单独创建本地虚拟环境；已有可用环境可以直接使用。不要复制另一个操作系统
的虚拟环境。运行时无需 `pip install -e .`，从包根目录执行模块即可。

```text
python -m venv .venv
```

Windows PowerShell 使用 `.venv\Scripts\python.exe`，Linux/macOS 使用
`.venv/bin/python`；下列命令中的 `python` 指你选定的解释器。

```text
python -m pip install "pytest>=8,<9"
python -B -m pytest tests/test_refund_decision_mvp.py tests/test_abcd_posttraining.py tests/test_abcd_rule_spec.py tests/test_local_cpu_delivery.py -q
```

只有安装 pytest 才可能访问包索引；已经安装时，运行检查不访问网络。
完整仓库仍使用原 `pytest` 范围，不把精简包测试通过说成全仓库测试通过。

## 构建/校验代码包

文件白名单是 `configs/local_bundle.json`。不依赖 Git 是否已跟踪：最新未提交
源码也会按清单进入快照，其状态写入 manifest；这不等于已经 commit 或 push。

```text
python -B -m scripts.build_local_bundle --output tmp/local_delivery/skill-annealing-cpu.zip
```

生成 ZIP、同名 `.manifest.json`，输出 ZIP 的 SHA256。包里另有
`bundle_manifest.json`，记录每个文件的长度和哈希。只对清单读取文件，拒绝
路径穿越、符号链接、敏感路径和异常大文件，检测常见秘密模式；模式检测不是
绝对安全保证，任何将来的外传仍需审阅具体清单和公司规定。

解压到一个新目录，在该目录中验证：

```text
python -B -m scripts.build_local_bundle --verify
```

校验覆盖 manifest 列出的文件；不证明该目录不存在后来添加的其他文件。
然后运行上面的 CPU 冒烟与测试。不要复用已有工作目录进行覆盖解压。

## 包含与不包含

包含退款核心、ABCD 数据/评估通用工具、开发规则解释器及对应合成测试。
ABCD builder 是历史 Product Defect 通用工具，保留供测试/复用，**不是新实验
的训练入口**；源数据读取/下载工具也不由冒烟流程调用。

不包含真实数据、原始对话、已生成模型答案、权重、tokenizer、历史远程控制器、
凭据、缓存或研究结果目录。完整 ABCD 审计需要完整研究仓库和独立提供的合法
输入，不能仅凭此包重放全部历史证据。MBPP 训练/沙箱流程也不在本包范围。

开发规则仍有部分未定义分支，不是独立认证金标准。当前没有可直接启动新正式
实验的训练包。完整项目导航见 [代码分区](docs/code_layout.md)。
