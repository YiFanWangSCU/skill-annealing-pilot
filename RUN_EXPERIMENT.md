# 一处完成：规则级曝光小探索

这是当前入口。旧 START_HERE/SERVER_START 保留作基础环境说明；旧 ABCD 15-run
计划不执行。本包已经包含新探索的数据生成、训练、推理、统计与恢复代码。
它是**单种子、模拟规则的探索**，不是已经跑通 A800 的正式论文结果。

## 要跑什么

相同订单案例、政策配置和标签下，比较 Full-only、Short-only、整段混合
Whole-mix、按稳定规则模块混合 Component-mix。**四组始终看到完整动态参数**。
程序评估原政策、未见参数组合、未见参数取值，并区分受修改影响/不受影响案例。
精确定义、限制和停止条件见 [实验协议](docs/rule_exposure_protocol.md)。

默认预算：128 个训练案例 × 4 政策 = 每组 512 行；单种子，每组 64 updates，
共 4 次 LoRA 训练。基础模型与 4 个适配器各评估 384 个提示，共 1920 次生成。
每阶段最多 2 小时；整个 campaign 累计最多 8 小时（非运行时间预测）。

## 1. 克隆与环境

~~~bash
git clone https://github.com/YiFanWangSCU/skill-annealing-pilot.git
cd skill-annealing-pilot
~~~

此公开仓库使用 HTTPS，无需 GitHub 登录、token 或 SSH 密钥。
已 clone 此仓库时先保留本地修改，再用 git pull --ff-only 更新。
如果之前 clone 的是私有版本，请保留旧目录，在新目录 clone 本公开仓库，
不要混合两边的独立 Git 历史或覆盖原实验产物。公司网络受限时见
[离线传输](docs/company_handoff.md)。

建议 Python 3.12 的公司批准独立环境，不修改共享环境。先由管理员/你自己选定
兼容 CUDA 驱动的 PyTorch 及匹配的 torchvision，再安装其余固定依赖：

~~~bash
python -m pip install -r configs/server_requirements.txt
python -m pip check
python -B -m pytest -q
~~~

本仓库所有默认测试仅用 CPU、合成数据和模拟子进程。
代码沿用 ms-swift 4.1.1 / Transformers 5.5.4；没有静默切换框架或联网替换模型。
清单也包含 Qwen 加载器依赖 qwen-vl-utils/decord，即使本轮只输入文本也需要。
它是主要版本固定清单，不是完整传递依赖锁；预检记录并约束阶段间核心依赖版本。
参考 [SWIFT 模型文档](https://swift.readthedocs.io/en/v4.0/BestPractices/Qwen3_5-Best-Practice.html)
及 [PyTorch 安装入口](https://pytorch.org/get-started/locally/)。

## 2. 先验一遍完整流程，不需要 GPU 或模型

~~~bash
python -B -m scripts.rule_exposure simulate --workdir outputs/rule_cpu_check
~~~

它生成同一份数据，用带明确标记的 oracle 假预测验证统计/报告链路。
查看 outputs/rule_cpu_check/simulation_summary/report.md。
其中所有结果都标为 SIMULATION，**不能当模型分数**，不能用该目录做正式 run。
重跑 CPU 检查请换新目录。

## 3. 设置模型和获分配的 GPU

~~~bash
export MODEL_DIR='<complete-local-Qwen3.5-4B-directory>'
export GPU_INDEX='<allocated-gpu-index>'
~~~

请替换两个占位符。模型须有本地 config、tokenizer 和完整 safetensors 分片；
代码核对 Qwen3.5 dense 类型并记录全文件哈希，但型号是否为 4B 仍需你确认。
没有远程源码信任或自动 Hub 下载。与原实验权重不同会形成独立模型身份。
现有 CUDA_VISIBLE_DEVICES 若与 GPU_INDEX 不一致会拒绝，不覆盖调度分配。
当前支持普通单进程 shell；不要套 torchrun、多卡或未适配的 Slurm 启动器。

## 4. 一条命令完整运行

先看计划（会生成数据，但不加载模型/占 GPU）：

~~~bash
python -B -m scripts.rule_exposure run --workdir outputs/rule_pilot --model "$MODEL_DIR" --gpu "$GPU_INDEX"
~~~

确认资源分配后，执行同一命令加 --execute：

~~~bash
python -B -m scripts.rule_exposure run --workdir outputs/rule_pilot --model "$MODEL_DIR" --gpu "$GPU_INDEX" --execute
~~~

依次自动进行：精确 tokenizer 预检 → 四组训练 → 基础模型与四个适配器推理 →
严格 JSON 评分 → 按案例配对统计 → Markdown/CSV/JSON 报告。
预检会读取并哈希完整模型，可能需要一点时间；不会只靠模型目录名认定身份。
token 超长、初始化不一致、训练步数不符、GPU 忙或进程失败都会停止。
不会杀其他人的任务、自动找卡、切换模型、增加预算或挑选更好的 checkpoint。

建议在公司允许的 tmux 会话中前台运行以避免 SSH 断开；本程序不自动创建调度任务。
新进程组受超时管理，只清理本程序自己启动的进程组。锁仅协调同一 checkout
里的本探索，不是整台共享服务器的资源调度器；资源分配仍由公司管理。

## 5. 看进度、恢复、拿结果

~~~bash
python -B -m scripts.rule_exposure status --workdir outputs/rule_pilot
~~~

详细日志保存在 outputs/rule_pilot/stages/*/worker.log；训练更新进度在各训练
阶段的 training_audit.json。这些是本地日志，不能原样上传。

再次执行相同 run 命令会校验并跳过已完成阶段。失败尝试不会被覆盖或静默重试。
只在环境问题已修复、且代码/配置/模型未变时，显式加 --retry-failed 重试失败阶段：

~~~bash
python -B -m scripts.rule_exposure run --workdir outputs/rule_pilot --model "$MODEL_DIR" --gpu "$GPU_INDEX" --execute --retry-failed
~~~

每阶段最多两个尝试，失败也扣累计时限。控制器被强制中断时，未完成阶段保守扣除
其预留时限。代码、数据、配置、模型位置、GPU 改变会拒绝复用；应新建 campaign，
保留旧记录并说明变更，不把它当原实验无缝续跑。低分不是重试理由。

全部完成后看：

- outputs/rule_pilot/summary/report.md：人可读结果。
- outputs/rule_pilot/summary/metrics.csv：各策略/提示条件指标。
- outputs/rule_pilot/summary/summary.json：配对区间、输入哈希和限制。

可以重新汇总，**不重新推理**：

~~~bash
python -B -m scripts.rule_exposure summarize --workdir outputs/rule_pilot
~~~

## 不需要来回拷什么

源码、协议和测试都在此分支，模拟数据由代码本地生成。权重直接使用公司本地目录。
不需要实验室服务器上的旧适配器、ABCD 原始对话或 MBPP 数据。
不上传公司数据、密钥、模型、原始生成或日志；需要回传时只审阅脱敏后的汇总。
这一轮只实现和 CPU 验证了闭环，真实 A800 上的框架兼容性仍需首次运行确认。
