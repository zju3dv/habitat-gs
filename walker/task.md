你现在在 Ubuntu 项目目录：

```bash
~/habitat-gs/walker
```

当前已经完成了 Mixamo FBX 的预处理，并生成了可用的人物动作资产：

```bash
assets/humans/walk_meters_inplace.npz
```

该文件已经检查通过：

```text
vertices: (250, 14232, 3)
faces: (28272, 3)
height: 1.7919356
horizontal displacement norm: 3.185e-06
OK: height looks reasonable.
OK: root motion looks removed / in-place.
```

这个 npz 表示一个 **meter scale、Y-up、in-place walking baked mesh animation**。
当前目标是继续实现一个可接入 Habitat-GS 的 **MeshHumanActor 系统**，先不直接修改 Habitat-GS 渲染主流程，先把人物运动、状态机、轨迹控制、OBJ 导出验证全部做好。

## 总目标

实现一个动态 mesh 行人模块，用来替代 Habitat-GS 原来的 Gaussian avatar。当前阶段先完成独立模块：

```text
baked mesh animation clip
+ trajectory-driven root motion
+ state machine
+ capsule proxy
+ OBJ sequence debug export
```

最终要求能够做到：

```text
1. 读取 baked walk / idle / turn 等 npz clip
2. 读取 trajectory.npy
3. 每个时间步根据 state_id 选择动作 clip
4. 根据 root pose 把人物 mesh 放到世界坐标
5. 输出当前帧 world_vertices / faces / id / capsule
6. 导出连续 OBJ 序列，能在 Blender 中看到人物沿轨迹移动
```

## 当前已有资产

已有：

```bash
assets/humans/walk_meters_inplace.npz
```

后续可能会添加：

```bash
assets/humans/idle_meters_inplace.npz
assets/humans/turn_left_meters_inplace.npz
assets/humans/turn_right_meters_inplace.npz
```

如果某个 clip 不存在，系统不要崩溃，应 fallback 到 walk 或 idle。

## 坐标约定

必须遵守：

```text
X/Z: horizontal plane
Y: up axis
unit: meter
heading: yaw rotation around Y axis
heading = 0 means the human faces / moves along +Z
```

trajectory 格式统一为：

```text
shape: [T, 7]
columns:
0: x
1: y
2: z
3: heading
4: speed
5: state_id
6: time
```

state_id 定义：

```text
0: idle
1: walk
2: slow_walk
3: turn_left
4: turn_right
5: stop
```

当前最少只要求支持：

```text
0 idle
1 walk
```

如果 idle clip 不存在，则 idle state 暂时使用 walk 的第一帧或 walk clip fallback。

## 需要新增的代码结构

请在项目中新增目录：

```bash
human_actor/
```

并实现以下文件：

```bash
human_actor/__init__.py
human_actor/clip.py
human_actor/trajectory.py
human_actor/mesh_human_actor.py
human_actor/mesh_human_manager.py
human_actor/obj_export.py
```

同时新增工具脚本：

```bash
tools/create_state_trajectory.py
tools/export_actor_obj_sequence.py
tools/inspect_human_actor.py
```

## 具体实现要求

### 1. `human_actor/clip.py`

实现 `BakedMeshClip` 类。

功能：

```python
class BakedMeshClip:
    def __init__(self, npz_path: str, name: str = None):
        ...

    def frame_id(self, sim_time: float) -> int:
        ...

    def vertices_at(self, sim_time: float) -> np.ndarray:
        ...

    def normals_at(self, sim_time: float) -> Optional[np.ndarray]:
        ...
```

读取字段：

```text
vertices
faces
face_uvs
normals optional
fps
bbox_min
bbox_max
```

要求：

* vertices dtype float32
* faces dtype int32
* 支持循环播放
* 如果 npz 缺字段，要给清晰 error message

### 2. `human_actor/trajectory.py`

实现：

```python
class HumanTrajectory:
    def __init__(self, npy_path: str, fps: int = 30):
        ...

    def sample(self, sim_time: float) -> dict:
        ...
```

返回：

```python
{
    "x": float,
    "y": float,
    "z": float,
    "heading": float,
    "speed": float,
    "state_id": int,
    "time": float,
}
```

要求：

* 如果 sim_time 超出范围，clamp 到最后一帧
* 如果 trajectory shape 不是 [T,7]，报错

### 3. `human_actor/mesh_human_actor.py`

实现 `MeshHumanActor`。

初始化参数建议：

```python
class MeshHumanActor:
    def __init__(
        self,
        actor_id: int,
        name: str,
        clips: Dict[str, BakedMeshClip],
        trajectory: HumanTrajectory,
        fallback_clip: str = "walk",
        capsule_radius: float = 0.35,
        capsule_height: float = 1.70,
    ):
        ...
```

核心方法：

```python
def state_to_clip_name(self, state_id: int) -> str:
    ...

def root_pose_at(self, sim_time: float) -> np.ndarray:
    # return [x, y, z, heading]

def world_vertices_at(self, sim_time: float) -> np.ndarray:
    ...

def world_normals_at(self, sim_time: float) -> Optional[np.ndarray]:
    ...

def mesh_at(self, sim_time: float) -> dict:
    # return vertices, faces, face_uvs, normals, actor_id, name

def capsule_at(self, sim_time: float) -> dict:
    ...
```

root transform 公式：

```text
Y-up yaw rotation:

R = [
  [ cos, 0, sin],
  [   0, 1,   0],
  [-sin, 0, cos]
]
```

然后：

```python
verts_world = verts_local @ R.T + [x, y, z]
```

capsule 返回：

```python
{
    "actor_id": actor_id,
    "name": name,
    "center": np.array([x, y + height * 0.5, z], dtype=np.float32),
    "radius": radius,
    "height": height,
    "heading": heading,
}
```

### 4. `human_actor/mesh_human_manager.py`

实现 `MeshHumanManager`，支持多个人。

```python
class MeshHumanManager:
    def __init__(self, actors: List[MeshHumanActor]):
        ...

    def meshes_at(self, sim_time: float) -> List[dict]:
        ...

    def capsules_at(self, sim_time: float) -> List[dict]:
        ...
```

### 5. `human_actor/obj_export.py`

实现 OBJ 导出函数：

```python
def write_obj(out_path: str, vertices: np.ndarray, faces: np.ndarray):
    ...
```

以及支持导出多个 actor 到一个 OBJ：

```python
def write_multi_actor_obj(out_path: str, meshes: List[dict]):
    ...
```

注意多个 actor 合并 OBJ 时，face index 要正确 offset。

## 工具脚本要求

### A. `tools/create_state_trajectory.py`

生成一个 walk-idle-walk 轨迹，用来测试状态机。

命令示例：

```bash
python3 tools/create_state_trajectory.py \
  --out assets/humans/target_walk_idle_walk_traj.npy \
  --fps 30 \
  --start_x 0 \
  --start_y 0 \
  --start_z 2 \
  --heading 0 \
  --walk_speed 0.8
```

轨迹设计：

```text
0-3s: walk, z forward, state_id=1
3-5s: idle, position fixed, state_id=0
5-8s: walk, continue z forward, state_id=1
```

输出 shape:

```text
[240, 7]
```

### B. `tools/export_actor_obj_sequence.py`

读取 clip 和 trajectory，导出连续 OBJ 序列。

命令示例：

```bash
python3 tools/export_actor_obj_sequence.py \
  --walk_clip assets/humans/walk_meters_inplace.npz \
  --trajectory assets/humans/target_walk_idle_walk_traj.npy \
  --out_dir assets/debug_objs/target_walk_idle_walk \
  --num_frames 240 \
  --fps 30
```

如果传入 idle clip：

```bash
python3 tools/export_actor_obj_sequence.py \
  --walk_clip assets/humans/walk_meters_inplace.npz \
  --idle_clip assets/humans/idle_meters_inplace.npz \
  --trajectory assets/humans/target_walk_idle_walk_traj.npy \
  --out_dir assets/debug_objs/target_walk_idle_walk \
  --num_frames 240 \
  --fps 30
```

如果没有 idle clip，idle 状态 fallback 到 walk 第一帧或者 walk clip。

### C. `tools/inspect_human_actor.py`

打印 actor 在不同时间的状态：

```bash
python3 tools/inspect_human_actor.py \
  --walk_clip assets/humans/walk_meters_inplace.npz \
  --trajectory assets/humans/target_walk_idle_walk_traj.npy
```

需要输出：

```text
t=0.0 state=walk position=...
t=3.5 state=idle position=...
t=6.0 state=walk position=...
bbox world min/max
capsule center/radius/height
```

## 测试命令

完成后，请保证以下命令能跑通：

```bash
cd ~/habitat-gs/walker

python3 tools/create_state_trajectory.py \
  --out assets/humans/target_walk_idle_walk_traj.npy \
  --fps 30 \
  --start_x 0 \
  --start_y 0 \
  --start_z 2 \
  --heading 0 \
  --walk_speed 0.8

python3 tools/inspect_human_actor.py \
  --walk_clip assets/humans/walk_meters_inplace.npz \
  --trajectory assets/humans/target_walk_idle_walk_traj.npy

python3 tools/export_actor_obj_sequence.py \
  --walk_clip assets/humans/walk_meters_inplace.npz \
  --trajectory assets/humans/target_walk_idle_walk_traj.npy \
  --out_dir assets/debug_objs/target_walk_idle_walk \
  --num_frames 240 \
  --fps 30
```

运行完成后应该生成：

```bash
assets/debug_objs/target_walk_idle_walk/human_0000.obj
assets/debug_objs/target_walk_idle_walk/human_0030.obj
assets/debug_objs/target_walk_idle_walk/human_0105.obj
assets/debug_objs/target_walk_idle_walk/human_0180.obj
```

这些 OBJ 用 Blender 打开时，应能看到：

```text
0-3s 人物沿 +Z 方向移动
3-5s 人物停在原地
5-8s 人物继续沿 +Z 方向移动
```

## 代码质量要求

* 不要依赖 Habitat-GS 主代码。
* 不要修改已有 Habitat-GS 渲染逻辑。
* 所有新增模块先独立运行。
* 所有路径通过参数传入，不要写死绝对路径。
* 代码要有清晰错误提示。
* numpy dtype 尽量保持 float32 / int32。
* 保持当前坐标约定：X/Z 水平，Y 竖直。
* 输出日志要清晰，方便我判断是否成功。

## 最终交付

请完成：

```text
1. human_actor 模块
2. 三个 tools 脚本
3. 能跑通的测试命令
4. README 或终端输出说明，告诉我如何检查 OBJ 序列
```

当前任务只做到 **mesh human actor + trajectory + state machine + OBJ debug export**，不要直接接入 Habitat-GS renderer。等这个模块验证正确后，再进行下一步：mesh renderer + depth composite + Habitat-GS observation integration。
