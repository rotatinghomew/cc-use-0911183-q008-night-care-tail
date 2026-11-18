# 夜间照护共同突发负荷业务服务

现有能力是事件区间查询、平均占用描述和当夜资质盘点。需求时间、到场时间、结束时间分别保存；未结束的事件只累计窗口内已观察部分并保留 open_end。事件与呼叫分开，资质人数是重叠集合而非可相加的容量。`POST /api/event`、`POST /api/roster` 登记事实；`GET /api/load?start=2026-10-01T22:00&end=2026-10-02T06:00` 返回描述负荷。

在此之上，本服务提供面向应急储备决策的共同发生分析：以整夜为不可拆分的联合样本重建时间线、重放派班队列、比较候选技能组合，并给出统计保障边界与可追溯复盘。系统只提供分析证据，不替代现场派班。

## 方法学边界

分析遵循以下口径，任何结论都在该口径下解释：

1. **共同发生而不是独立相加。** 每个夜晚是一个联合样本：同夜各类需求的同时占用、人员接连参与不同服务的关系在分钟网格上一起重放。不会把各服务类别的极端负荷分别相加，也不假定类别彼此独立。跨夜不确定性通过整夜重采样（bootstrap）表达。
2. **三类时间分开。** 需求发生（start/呼叫）、等到人员（arrived/派班）、实际占用（arrived→end）分别建模；尚未结束的服务不会被当成短时完成。
3. **观察窗口之外不丢失。** 默认窗口为 22:00 至次日 06:00。跨窗口的片段被截断计入窗口内部分，并以 `carry_in`、`censored`、`open_end` 标记保留，不会悄悄消失。
4. **不确定性给出区间而非单点。** 四象限口径同时呈现：开放结束按观察截断（observed）或按同技能 0.9 分位数补全（imputed）；呼叫按事件合并（episode）或按合并间隔拆分（calls，默认 5 分钟）。储备判断取保守象限 `imputed-calls`，并可对比其余象限看结论对假设的敏感程度。
5. **一个人的能力不可重复使用。** 重放中人员以技能组合参与，一次只承接一个服务；双证人员被普通照护占用时，其转运能力同时不可用。冗余以逐分钟空闲人力度量，而不是把各资质人数相加。
6. **证据边界诚实。** 风险比例同时给 Clopper–Pearson 精确区间和整夜重采样区间；零失败夜晚给出单侧上界（约 3/n，60 夜约为 4.9%），小样本不会输出零风险。方案若在历史最忙共同突发夜仍不达标，结果标记为含外推，不能声称已证明极低风险。
7. **普通夜与突发夜同一解释。** 突发夜按历史联合峰值分布的 0.8 分位数划分，两组使用同一重放与目标口径，仅样本不同。

## 分析接口

| 接口 | 用途 |
| --- | --- |
| `GET /api/timeline?night=2026-10-01&call_policy=calls&merge_gap=0&open_end_policy=imputed` | 整夜时间线、占用网格、最忙时段及其共同事件构成 |
| `GET /api/replay?night=2026-10-01&mode=feasible` | 当夜队列重放；`greedy` 为固定优先级派班，`feasible` 搜索是否存在满足响应目标的可行派班 |
| `POST /api/evaluate`，body `{"plan_ids": [...], "mode": "greedy", "replicates": 1000}` | 候选方案在全部历史夜晚的四象限评估、风险边界、普通/突发分组与冗余 |
| `POST /api/reconcile-night`，body `{"night": "...", "plan_id": "..."}` | 单夜复盘并持久化，引用负荷假设编号 |
| `POST /api/reconcile-period` | 一段时期的批量复盘与归因计数 |

复盘归因允许同一夜同时带有多个标签，不把未达标一概归因于需求增长：

* `demand_beyond_distribution`：当夜联合峰值超出由其他历史夜晚构成的占用包络；
* `staffing_shortfall`：实际当班人数或技能组合低于已采用方案（未按计划在岗）；
* `capability_allocation`：在岗满足方案、贪心派班超时，但最优搜索存在达标派班（分配问题）；
* `capacity_insufficient`：在岗满足方案，但最优搜索确认无解（能力总量不足）。

候选方案（`fixtures/plans.json`）与负荷假设（`fixtures/assumptions.json`，含窗口、口径、分位数、置信度、响应目标）均带编号；复盘记录保存所引用的假设编号与方案编号，支持日后区分需求分布变化、缺岗和能力分配三类缺口。

## 启动与核对

使用 Python 3.11 或更高版本。现有服务采用标准库运行；数值分析所需的固定版本工具列在 requirements.txt，部署准备时运行 `python3 -m pip install -r requirements-build.txt -r requirements.txt`，构建前端和打包后端的固定版本见 requirements-build.txt；依赖准备完毕后，测试、构建和运行均可断网进行。无需真实个人信息、凭据或远程数据源。

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q nightload tests
python3 -m build --wheel --no-isolation
python3 -m nightload check
python3 -m nightload demo
python3 -m nightload timeline --night 2026-10-01 --open-end-policy imputed --call-policy calls
python3 -m nightload replay --night 2026-10-01 --mode feasible
python3 -m nightload evaluate --plans plan-baseline,plan-flex-four,plan-specialist-four --replicates 1000
python3 -m nightload reconcile --night 2026-07-14 --plan plan-baseline
python3 -m nightload load --database /tmp/nightload-business.sqlite3 --input examples/operations.json
python3 -m nightload serve --database /tmp/nightload-business.sqlite3
```

服务默认监听本机 8765 端口，`--port` 可指定其他端口。健康检查为 `GET /health`，`GET /api/summary` 返回当前已有业务汇总。未指定数据库时，demo 和 check 使用隔离的临时数据库；serve 若需保留修改须传入数据库路径。已有数据库不重复装载同一类样例。examples/operations.json 是较大的一组历史运营资料（60 个夜班、405 场事件、535 次呼叫、180 条当班记录），可用 load 命令在事务内载入；按类型和编号更正同名记录，未出现的记录保持不变。所有资料采用固定来源生成且不包含对未来分析结论的预设值。

## 数据与运行边界

fixtures 中均为本地合成业务资料，员工、家庭和院区编号不对应真实对象。JSON 里的空值表示未记录，不代表零；日期采用 ISO 格式。金额以分保存，工时以分钟保存。每类记录有独立编号，写入同编号用于业务更正，revision 用于并发修改核对。

业务状态由 SQLite 保存，记录批量更正在单一事务内提交。接口使用 UTF-8 JSON，未知资源返回 404，非法数据返回 400，版本冲突返回 409。服务不进行临床诊断，也不自动改变实际服务安排。测试使用临时数据库和系统分配的空闲端口，彼此隔离，不访问公网。运行日志、数据库和依赖缓存不应提交到仓库。
