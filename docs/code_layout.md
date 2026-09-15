# 代码导航

当前服务器分支的唯一实验入口是 RUN_EXPERIMENT.md。
它包含一套自足的模拟规则小探索；历史研究证据库仍在原研究仓库。

| 路径 | 职责 |
| --- | --- |
| scripts/rule_exposure.py | 数据准备、CPU 模拟、完整 GPU 运行、恢复、汇总、状态 |
| scripts/rule_exposure_worker.py | tokenizer 预检、SWIFT 训练、Transformers 推理 |
| scripts/rule_exposure_callback.py | 实际更新步数、LoRA 起始一致性及训练后变化记录 |
| skill_annealing/rule_exposure/data.py | 独立模拟规则、配对数据、曝光掩码、确定性标签及哈希 |
| skill_annealing/rule_exposure/runtime.py | 运行环境、训练参数和阶段产物核验 |
| skill_annealing/rule_exposure/analysis.py | 严格评分、政策变化分析、案例级配对区间及报告 |
| configs/rule_exposure_pilot.json | 当前探索预算和超参数 |
| docs/rule_exposure_protocol.md | 设计、对比、统计及研究限制 |
| tests/test_rule_exposure.py | 无 GPU 的规则、数据、恢复与后端接口模拟测试 |

旧 skill_annealing/refund_decision 与 abcd_posttraining 工具保留供复用；
它们不生成本轮新探索数据，也不自动认证真实业务标签。
scripts/local_cpu_smoke.py 和 START_HERE.md 是独立的旧 CPU 核心检查；
scripts/server_smoke.py 是可选两步环境检查，不是完整实验入口。
如果仅解压 configs/local_bundle.json 的 CPU 子包，可能不含上表的新实验文件；
应通过 Git 分支或 configs/server_bundle.json 获取完整服务器执行包。

outputs/、tmp/、data/、models/ 存放本地产物，不整体打包或上传。
打包只读取明确的源码白名单；bundle_manifest.json 校验列出文件的字节，
不认证额外文件安全。真实数据、模型、原始生成和日志均不在此源码包中。
