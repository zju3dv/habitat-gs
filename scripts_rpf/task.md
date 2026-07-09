你是一个严谨的机器人仿真代码助手。请先阅读当前仓库结构，然后实现一个最小可运行版本的 robot person following expert，用于离线生成 expert trajectory 数据。

# 目标

实现一个最简单的 **Forecast-Aware Formation Expert**。

它在每个 timestep 读取仿真中的 privileged GT 信息：

* robot 当前位姿
* target 当前与未来轨迹
* other pedestrians 当前与未来轨迹
* scene / walkable area 信息，如果当前仓库已有接口就使用；没有就先留 stub

然后 expert 直接生成机器人未来 3 秒的局部跟随轨迹，并输出当前一步 `cmd_vel = (v, omega)`。

核心目标是跑通数据生成流程：

```text
fixed episode
→ pedestrians follow predefined paths
→ expert controls robot online
→ save RGB / robot state / expert trajectory / cmd_vel / labels
```

# 不要做的事情

请不要实现复杂 RL。
请不要调用 VLM。
请不要实现 branch action scoring。
请不要采样大量候选轨迹再打分。
请不要引入大型模型。

第一版只需要规则式 expert，把流程跑通。

# Expert 设计

expert 每一帧根据目标未来轨迹生成目标中心参考点：

```text
p_ref(k) = p_target(k) - d(k) * target_forward(k) + y(k) * target_side(k)
```

其中：

* `d(k)` 是跟随距离
* `y(k)` 是侧向偏移
* 默认 `d = 2.0m`
* 默认 `y = +0.8m` 或 `-0.8m`
* `d` 限制在 `[1.5, 3.0]`
* `y` 限制在 `[-1.2, 1.2]`

# 第一版 expert 规则

只实现 4 条规则：

## 规则 1：正常跟随

默认保持目标侧后方：

```text
d = 2.0
y = preferred_side * 0.8
```

其中 `preferred_side = +1` 表示左后方，`preferred_side = -1` 表示右后方。

## 规则 2：同侧有迎面行人

如果未来 3 秒内，机器人当前 preferred side 一侧有行人迎面靠近目标侧后方区域：

```text
y -> 0.0
d -> 2.3
```

含义：机器人并回目标正后方。

## 规则 3：目标将被遮挡

如果未来 3 秒内有行人进入 robot-target line-of-sight corridor：

```text
如果遮挡来自左侧，y 向右侧调整
如果遮挡来自右侧，y 向左侧调整
```

第一版可以用简单几何近似：

* 计算 pedestrian 到 robot-target 线段的距离
* 如果距离小于 `0.5m`，认为有遮挡风险
* 根据 pedestrian 相对 target 的左右方向调整 `y`

## 规则 4：人群过近或目标停止

如果目标速度很低，或者目标附近 1.2m 内有多个行人：

```text
d -> min(d + 0.5, 3.0)
v_cmd 降低
```

# Safety projection

生成未来 3 秒参考轨迹后，对每个点检查和其他行人的距离。

参数：

```text
collision_radius = 0.4
personal_radius = 0.8
social_radius = 1.2
```

如果某个参考点距离某个 pedestrian 小于 `personal_radius`，把这个点沿远离 pedestrian 的方向推开：

```python
p_ref = p_ref + beta * normalize(p_ref - p_ped)
```

第一版 `beta = 0.3`。

如果推开后仍然小于 `collision_radius`，标记该 timestep 为 unsafe，并降低速度。

# Trajectory smoothing

对参考点做简单平滑：

* moving average
* 或 cubic spline，如果仓库已有 scipy 可以用
* 第一版优先用 moving average，减少依赖

# Controller

实现一个简单 pure pursuit / proportional controller。

输入：

* robot 当前位姿
* expert local trajectory

输出：

```text
cmd_vel = (v, omega)
```

参数：

```text
v_max = 1.0
omega_max = 1.2
lookahead_index = 3
k_v = 0.8
k_omega = 1.5
```

控制逻辑：

```python
lookahead = trajectory[lookahead_index]
dx, dy = lookahead in robot frame
distance = sqrt(dx^2 + dy^2)
heading_error = atan2(dy, dx)

v = clip(k_v * distance, 0, v_max)
omega = clip(k_omega * heading_error, -omega_max, omega_max)
```

如果当前 timestep 被标记 unsafe：

```python
v *= 0.5
```

# 数据保存

每个 timestep 保存一个 json record：

```json
{
  "timestamp": 0.0,
  "episode_id": "xxx",
  "scenario_type": "side_follow_with_oncoming_pedestrian",

  "robot_state": {
    "x": 0.0,
    "y": 0.0,
    "theta": 0.0,
    "v": 0.0,
    "omega": 0.0
  },

  "target_id": 1,

  "expert_trajectory_world": [
    [x1, y1, theta1],
    [x2, y2, theta2]
  ],

  "expert_trajectory_local": [
    [dx1, dy1],
    [dx2, dy2]
  ],

  "expert_action": {
    "v": 0.5,
    "omega": 0.1
  },

  "labels": {
    "visibility_risk": 0.0,
    "social_risk": 0.0,
    "unsafe": false,
    "collision": false,
    "target_lost": false
  }
}
```

RGB 图像如果当前仓库已有渲染接口，就保存路径：

```json
"rgb_path": "frames/episode_xxx/frame_000123.png"
```

如果暂时没有图像接口，保留字段为 null，不要阻塞 expert 流程。

# 希望新增的文件

请根据仓库结构自行决定文件位置。推荐结构：

```text
scripts_rpf/rpf_expert/
    __init__.py
    formation_expert.py
    trajectory_utils.py
    social_risk.py
    controller.py
    validator.py
    logger.py

scripts/
    run_minimal_expert_episode.py
```

如果仓库已经有类似模块，请复用已有结构，不要重复造轮子。

# 核心类设计

请实现：

```python
@dataclass
class Pose2D:
    x: float
    y: float
    theta: float

@dataclass
class AgentState:
    agent_id: int
    pose: Pose2D
    velocity: tuple[float, float]

@dataclass
class ExpertConfig:
    horizon_sec: float = 3.0
    dt: float = 0.2
    default_distance: float = 2.0
    preferred_lateral_offset: float = 0.8
    preferred_side: int = 1
    min_distance: float = 1.5
    max_distance: float = 3.0
    max_lateral_offset: float = 1.2
    collision_radius: float = 0.4
    personal_radius: float = 0.8
    social_radius: float = 1.2
    v_max: float = 1.0
    omega_max: float = 1.2
```

核心 expert 类：

```python
class ForecastAwareFormationExpert:
    def __init__(self, config: ExpertConfig):
        ...

    def step(
        self,
        robot_state: AgentState,
        target_future: list[AgentState],
        pedestrians_future: dict[int, list[AgentState]],
    ) -> dict:
        """
        Return:
            expert_trajectory_world
            expert_trajectory_local
            cmd_vel
            labels
        """
```

# 最小 episode runner

`scripts/run_minimal_expert_episode.py` 先实现一个不依赖复杂仿真的 toy runner：

* 构造一个目标直线行走轨迹
* 构造一个迎面行人轨迹
* robot 初始在目标左后方
* 每一帧调用 expert.step()
* 用 cmd_vel 更新 robot pose
* 保存 jsonl
* 可选保存 top-down debug plot

这一步非常重要，用于验证 expert 逻辑。

toy episode 场景：

```text
target: 从 (0,0) 向 x 正方向走，速度 0.8 m/s
robot: 初始在 target 左后方，(-2.0, 0.8)
oncoming pedestrian: 从 (4.0, 1.0) 向 x 负方向走
```

期望行为：

```text
robot 初始保持左后方
当迎面行人进入左侧风险区域时，robot 的 y offset 收敛到 0
robot 回到目标后方
通过风险区域后可以逐渐恢复 preferred side
```

# Debug 输出

请在 toy runner 中输出：

```text
t
robot pose
target pose
d
y
cmd_vel
social_risk
visibility_risk
unsafe
```

并保存：

```text
outputs/minimal_expert_episode/episode.jsonl
outputs/minimal_expert_episode/topdown.png
```

# 代码质量要求

* 代码要模块化
* 每个函数加必要 docstring
* 避免过度工程化
* 避免硬编码过多路径
* 加基础单元测试或至少提供可运行 smoke test
* 运行脚本后应该能看到 jsonl 和 topdown plot
* 如果缺少依赖，请优先使用 numpy / matplotlib，不要引入重依赖

# 验收标准

运行：

```bash
python scripts/run_minimal_expert_episode.py
```

应该生成：

```text
outputs/minimal_expert_episode/episode.jsonl
outputs/minimal_expert_episode/topdown.png
```

episode.jsonl 中每一帧包含：

* robot_state
* expert_trajectory_world
* expert_trajectory_local
* expert_action
* labels

topdown.png 中应能看到：

* target trajectory
* robot trajectory
* oncoming pedestrian trajectory
* robot 在遇到同侧迎面行人时向目标后方收敛的行为

请先实现这个最小版本，保证能跑通，再考虑接入真实 Habitat-GS / Carla 接口。
