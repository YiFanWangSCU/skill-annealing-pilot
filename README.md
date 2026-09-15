# Skill Exposure Training — 完整小探索执行包

当前唯一运行入口：[RUN_EXPERIMENT.md](RUN_EXPERIMENT.md)。
本公开仓库包含数据生成、四组配对 LoRA 训练、基础模型/适配器推理、
严格评分、配对统计、报告和失败续跑，不需要从实验室另拷脚本。

## 这轮研究什么

让模型学习稳定的退款处理流程，同时始终提供会变化的当前政策参数。
比较 Full-only、Short-only、整段混合、逐模块混合，检验短提示下对新政策条件
的适应情况。它是单种子、模拟结构化规则的探索，**不是已验证的论文结论**。

默认四组各 64 updates；基础模型加四个训练端点，共 1920 次生成。
每阶段最多 2 小时，累计 worker 运行预算 8 小时；不是耗时预测。
详细设计见 [协议](docs/rule_exposure_protocol.md)，状态见
[pilot_registry.json](docs/pilot_registry.json)。

## 开始

~~~bash
git clone https://github.com/YiFanWangSCU/skill-annealing-pilot.git
cd skill-annealing-pilot
python -m pip install 'pytest>=8,<9'
python -B -m pytest -q
python -B -m scripts.rule_exposure simulate --workdir outputs/rule_cpu_check
~~~

模拟报告显式标记 SIMULATION，不是模型分数。
准备好文档要求的独立环境、本地模型及获分配 GPU 后，按 RUN_EXPERIMENT
执行一条 run --execute 命令，它会顺序完成全部实验阶段，不用逐阶段批准。

公司服务器传输和 Claude Code 可直接复制的任务见
[company_handoff.md](docs/company_handoff.md)。AI 助手先读 [CLAUDE.md](CLAUDE.md)。

## 范围和版本

这是采用全新 Git 历史的公开源码执行包，HTTPS clone 无需 GitHub 登录。
原私有仓库及其历史没有公开；旧研究证据库没有整体搬入。
源文件白名单与哈希见 bundle_manifest.json，运行版本记录 git rev-parse HEAD。
该清单校验列出的文件，不证明额外本地文件安全。

旧退款/ABCD 工具和 CPU/两步环境检查保留，不是本次数据来源或默认训练入口。
旧 15-run 方案不自动执行。权重、适配器、数据、原始生成、日志、服务器认证信息
均不上传；模拟数据在服务器本地生成。

已做 CPU 测试及模拟链路检查；真实 A800 的加载、反向传播与推理仍待首次运行验证。
