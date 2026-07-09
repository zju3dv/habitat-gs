你现在在 Ubuntu 项目目录：

```bash
~/habitat-gs/walker
```

当前已经完成了独立的 mesh human actor 模块，已有资产：

```bash
assets/humans/walk_meters_inplace.npz
```

该资产是：

```text
meter scale
Y-up
X/Z horizontal plane
in-place walking baked mesh animation
vertices: [T, N, 3]
faces: [F, 3]
face_uvs: [F, 3, 2]
normals: [T, N, 3]
```

当前目标是：**把 mesh human actor 接入 Habitat-GS 渲染/observation 流程中**，让 Habitat-GS 的 RGB/depth observation 中出现一个动态 mesh 行人，用来替代或绕开原来的 Gaussian avatar。

## 核心原则

不要大改 Habitat-GS 主架构。
不要删除原来的 Gaussian avatar 逻辑。
新增一个可开关模块：

```text
GS scene renderer
        ↓
MeshHumanActorManager
        ↓
MeshHumanRenderer
        ↓
Depth compositor
        ↓
Final RGB / depth observation
```

目标是让用户可以通过配置打开：

```yaml
mesh_humans:
  enabled: true
```

关闭后 Habitat-GS 行为保持原样。

---

## 当前已有模块

假设已有或需要保留以下模块：

```bash
human_actor/
├── __init__.py
├── clip.py
├── trajectory.py
├── mesh_human_actor.py
├── mesh_human_manager.py
└── obj_export.py
```

其中 `MeshHumanActor` 能提供：

```python
actor.mesh_at(sim_time)
actor.capsule_at(sim_time)
```

`mesh_at(sim_time)` 返回：

```python
{
    "vertices": np.ndarray,   # [N, 3], world coordinates, meter
    "faces": np.ndarray,      # [F, 3]
    "face_uvs": np.ndarray,   # [F, 3, 2]
    "normals": np.ndarray | None,
    "actor_id": int,
    "name": str,
}
```

---

# 任务目标

完成 Habitat-GS 集成，使得运行仿真时可以：

```text
1. Habitat-GS 正常渲染 GS 背景 RGB/depth
2. MeshHumanActor 根据 sim_time 更新世界坐标 mesh
3. MeshHumanRenderer 使用同一个 camera 渲染 human RGB/depth/id_mask
4. DepthCompositor 合成 GS 背景和 mesh human
5. 最终 observation["rgb"] 里能看到动态行人
6. observation["depth"] 能反映行人深度
7. 可选输出 observation["human_id_mask"]
```

---

# 非常重要：先定位 Habitat-GS 渲染入口

请先在代码中搜索：

```bash
grep -R "gaussian_avatars\|canonical_gs\|driver.pkl\|render.*depth\|get_sensor_observations\|observations\|rgb" -n .
```

需要找到 Habitat-GS 生成 RGB/depth observation 的位置。

不要假设文件名。
先阅读当前项目结构，再决定接入点。

最终请在代码注释或 README 中说明你接入的位置，例如：

```text
Integrated mesh human overlay after GS RGB/depth rendering and before returning sensor observation.
```

---

# 新增模块

请新增：

```bash
human_actor/rendering/
├── __init__.py
├── camera.py
├── simple_mesh_renderer.py
└── depth_compositor.py
```

## 1. `camera.py`

实现一个相机数据结构：

```python
@dataclass
class CameraParams:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_T_camera: np.ndarray  # [4, 4]
    camera_T_world: np.ndarray  # [4, 4]
    near: float = 0.01
    far: float = 100.0
```

并实现工具函数：

```python
def build_camera_from_habitat_sensor(...):
    ...
```

要求：

* 尽量从 Habitat-GS / Habitat-Sim 当前 sensor spec 中读取 intrinsics/extrinsics。
* 如果当前代码没有明确 fx/fy，则根据 HFOV 计算：

```python
fx = 0.5 * width / tan(0.5 * hfov)
fy = fx
cx = width / 2
cy = height / 2
```

* 必须打印一次 camera debug 信息：

```text
width, height, fx, fy, cx, cy
camera position
camera forward direction
```

---

## 2. `simple_mesh_renderer.py`

先实现一个可运行的 mesh renderer。优先级：

```text
1. 如果项目已有 nvdiffrast / pytorch3d，可以优先使用
2. 如果没有，就实现一个 numpy / torch 简单 z-buffer triangle rasterizer
```

当前阶段不要求照片级材质，先用固定颜色渲染 human：

```text
target actor: green-ish / red-ish
other actors: blue-ish / gray-ish
```

必须输出：

```python
rgb_human: np.ndarray      # [H, W, 3], uint8 or float32
depth_human: np.ndarray    # [H, W], float32, metric depth
id_mask: np.ndarray        # [H, W], int32, 0 means background
```

Renderer 接口：

```python
class SimpleMeshRenderer:
    def __init__(self, width: int, height: int):
        ...

    def render(self, meshes: list[dict], camera: CameraParams) -> dict:
        return {
            "rgb": rgb_human,
            "depth": depth_human,
            "id_mask": id_mask,
        }
```

投影约定：

```text
mesh vertices are in Habitat-GS world coordinates
convert world -> camera coordinates
project to pixels using fx/fy/cx/cy
use z-buffer to keep nearest triangle
```

注意：

* 要处理 camera/world 坐标系差异。
* 如果第一次渲染看不到人，请提供 debug 工具打印 projected bbox。
* 对每个 actor 输出 projected 2D bbox：

```text
actor_id=1 projected bbox: xmin, ymin, xmax, ymax, valid_vertices=...
```

---

## 3. `depth_compositor.py`

实现：

```python
def composite_rgbd(
    rgb_gs: np.ndarray,
    depth_gs: np.ndarray,
    rgb_human: np.ndarray,
    depth_human: np.ndarray,
    id_mask: np.ndarray,
) -> dict:
    ...
```

合成逻辑：

```python
valid_human = id_mask > 0
human_front = depth_human < depth_gs
visible = valid_human & human_front

rgb_final = rgb_gs.copy()
rgb_final[visible] = rgb_human[visible]

depth_final = depth_gs.copy()
depth_final[visible] = depth_human[visible]

id_mask_final = np.zeros_like(id_mask)
id_mask_final[visible] = id_mask[visible]
```

注意：

* 如果 Habitat-GS depth 是 inverse depth / normalized depth，要先转换或适配。
* 如果无法确认 depth 单位，请写 debug 输出 depth min/max。
* 必须保证原始 rgb/depth shape 不被破坏。

---

# 新增配置

新增配置文件示例：

```yaml
mesh_humans:
  enabled: true
  debug: true
  actors:
    - actor_id: 1
      name: target
      clips:
        walk: assets/humans/walk_meters_inplace.npz
      trajectory: assets/humans/target_walk_idle_walk_traj.npy
      fallback_clip: walk
      capsule_radius: 0.35
      capsule_height: 1.70
```

如果项目不用 yaml，而是 json/config class，请适配当前项目风格。

要求：

* 如果 `mesh_humans.enabled=false`，完全不影响原 Habitat-GS。
* 如果 `enabled=true`，加载 actor manager。
* 如果 clip 或 trajectory 不存在，给清晰错误。

---

# 集成位置

在 Habitat-GS 生成 observation 后加一层：

伪代码：

```python
obs = original_render_observation(...)

if mesh_humans_enabled:
    sim_time = get_current_sim_time_or_frame_id() / fps

    meshes = mesh_human_manager.meshes_at(sim_time)

    camera = build_camera_from_current_sensor(...)

    human_render = mesh_renderer.render(meshes, camera)

    composed = composite_rgbd(
        rgb_gs=obs["rgb"],
        depth_gs=obs["depth"],
        rgb_human=human_render["rgb"],
        depth_human=human_render["depth"],
        id_mask=human_render["id_mask"],
    )

    obs["rgb"] = composed["rgb"]
    obs["depth"] = composed["depth"]
    obs["human_id_mask"] = composed["id_mask"]
```

如果当前 observation key 不是 `"rgb"` / `"depth"`，请适配真实 key，并在 README 里说明。

---

# Debug 工具

新增：

```bash
tools/debug_render_mesh_human_in_habitat.py
```

功能：

```text
1. 加载一个 Habitat-GS scene
2. 加载一个 mesh human actor
3. 固定 robot camera pose
4. 渲染 GS background
5. 渲染 mesh human
6. depth composite
7. 保存图片：
   outputs/debug_gs_rgb.png
   outputs/debug_human_rgb.png
   outputs/debug_composed_rgb.png
   outputs/debug_human_id_mask.png
   outputs/debug_depth_gs.npy
   outputs/debug_depth_human.npy
```

如果直接接完整 Habitat-GS 太复杂，先做一个最小 debug hook：

```text
在当前项目已有 demo / render script 上加 mesh_human overlay 参数。
```

---

# 测试目标

请保证至少能跑通一个命令，类似：

```bash
python3 tools/debug_render_mesh_human_in_habitat.py \
  --scene_config <existing_habitat_gs_scene_config> \
  --walk_clip assets/humans/walk_meters_inplace.npz \
  --trajectory assets/humans/target_walk_idle_walk_traj.npy \
  --out_dir outputs/mesh_human_debug
```

如果当前项目的 scene config 参数不同，请根据实际 repo 写真实命令。

最终输出中必须有：

```text
debug_gs_rgb.png
debug_human_rgb.png
debug_composed_rgb.png
debug_human_id_mask.png
```

---

# 最小验收标准

完成后必须满足：

```text
1. 原 Habitat-GS 不开 mesh_humans 时能正常运行
2. 开 mesh_humans 后，RGB observation 中能看到 mesh 人
3. 人的位置随 trajectory 变化
4. 人能正确遮挡背景中更远的 GS 像素
5. depth_final 中人的深度比背景近
6. human_id_mask 中人物区域为 actor_id
7. 如果人不在相机视野内，不崩溃
8. Debug 输出 projected bbox / depth minmax
```

---

# 暂时不做的事情

当前任务不要做：

```text
1. 不要做复杂 PBR 材质
2. 不要做真实光照一致性
3. 不要接物理引擎
4. 不要改 Habitat NavMesh
5. 不要训练任何模型
6. 不要删除 Gaussian avatar
7. 不要大规模重构 Habitat-GS
```

当前只做：

```text
mesh human rendering + depth composition + Habitat-GS observation integration
```

---

# 代码质量要求

* 所有新增功能可开关。
* 路径通过配置或命令行传入，不要写死绝对路径。
* 尽量少侵入 Habitat-GS 原始代码。
* 每个集成点加注释。
* 对 shape、dtype、depth range 做 assert 或 debug print。
* 如果发现 camera 坐标系不一致，请新增明确的 conversion 函数，不要到处写 magic transform。
* 输出 README，说明：

  * 修改了哪些文件
  * 新增了哪些文件
  * 如何运行 debug
  * 如果人看不到，如何检查 projected bbox / depth / camera pose

---

# 最终交付

请完成：

```text
1. Mesh human renderer
2. Depth compositor
3. Habitat-GS observation integration hook
4. Config support
5. Debug render script
6. README / usage
```

当前阶段的目标只有一个：

**在 Habitat-GS 渲染结果中看到一个动态 mesh 行人，并且 RGB/depth/id_mask 合成正确。**
