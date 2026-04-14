#!/usr/bin/env python3

"""
Interactive Habitat-GS viewer with RGB/depth rendering, avatar debugging,
physics stepping, and object manipulation.

Usage:
    python gaussian_viewer.py --input /path/to/3dgs/file.ply
"""

import argparse
import ctypes
from enum import Enum
import math
import os
import string
import sys
import time
from typing import Any, Dict, Optional

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import magnum as mn
import numpy as np
from magnum import shaders, text
from magnum.platform.glfw import Application
from matplotlib import cm

import habitat_sim
from habitat_sim import physics
from habitat_sim.logging import LoggingContext, logger
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.utils.settings import default_sim_settings, make_cfg


class RenderMode(Enum):
    """Rendering mode enumeration."""
    RGB = "RGB"
    DEPTH = "DEPTH"


class MouseMode(Enum):
    LOOK = 0
    GRAB = 1


class GaussianViewer(Application):
    """Interactive viewer for 3D Gaussian Splatting with RGB and Depth rendering."""

    # Display text settings
    MAX_DISPLAY_TEXT_CHARS = 512
    TEXT_DELTA_FROM_CENTER = 0.49
    DISPLAY_FONT_SIZE = 16.0

    def __init__(
        self, 
        gaussian_file: Optional[str], 
        window_width: int = 800, 
        window_height: int = 600,
        depth_clip_min_percentile: float = 1.0,
        depth_clip_max_percentile: float = 99.0,
        dataset: Optional[str] = None,
        scene: Optional[str] = None,
        enable_physics: bool = True,
        start_time: Optional[float] = None,
        time_rate: float = 1.0,
        autoplay: Optional[bool] = None,
    ) -> None:
        self.gaussian_file = gaussian_file
        self.dataset = dataset
        self.scene = scene
        self.enable_physics = enable_physics

        # Initialize logging early so INFO logs in initialization are visible
        LoggingContext.reinitialize_from_env()
        logger.setLevel("INFO")

        # Setup window configuration
        configuration = self.Configuration()
        configuration.title = "Habitat-GS Viewer"
        configuration.size = (window_width, window_height)
        Application.__init__(self, configuration)
        
        self.fps: float = 60.0

        # Time controls for simulator Gaussian time, used by animated assets.
        self.current_time: float = start_time if start_time is not None else 0.0
        self.time_step: float = 0.0
        self.time_rate: float = time_rate
        self.time_max: Optional[float] = None
        self.time_playing: bool = autoplay if autoplay is not None else False
        self.loop_time: bool = True
        self.user_time_override = start_time is not None
        self.user_autoplay_override = autoplay is not None
        self.show_avatar_proxies: bool = False
        self.debug_nav_collision: bool = False
        self._nav_filter_log_interval: float = 0.25
        self._last_nav_filter_log: float = 0.0
        self._last_action_name: Optional[str] = None
        self._orig_move_filter_fn = None

        self.debug_bullet_draw = False
        self.contact_debug_draw = False
        self.semantic_region_debug_draw = False
        self.debug_semantic_colors: Dict[int, mn.Color4] = {}
        self.cached_urdf = ""

        self.mouse_interaction = MouseMode.LOOK
        self.mouse_grabber: Optional["MouseGrabber"] = None
        self.previous_mouse_point: Optional[mn.Vector2i] = None

        self.simulating = bool(enable_physics)
        self.simulate_single_step = False

        # NOTE: Starting with DEPTH mode is COMPULSORY for depth rendering pipeline to work properly
        self.render_mode = RenderMode.DEPTH

        # Depth visualization settings
        self.depth_min = 0.1
        self.depth_max = 10.0
        # Use Spectral_r (reversed) so near is red, far is blue
        self.colormap = cm.get_cmap('Spectral')
        
        # Clip percentiles for depth filtering
        self.depth_clip_min_percentile = depth_clip_min_percentile
        self.depth_clip_max_percentile = depth_clip_max_percentile
        
        # Depth shader for visualization
        self.depth_shader: mn.shaders.FlatGL2D = None
        self.depth_mesh: mn.gl.Mesh = None
        self.depth_texture: mn.gl.Texture2D = None

        # Movement controls
        key = Application.Key
        self.pressed = {
            key.UP: False,
            key.DOWN: False,
            key.LEFT: False,
            key.RIGHT: False,
            key.A: False,
            key.D: False,
            key.S: False,
            key.W: False,
            key.X: False,
            key.Z: False,
        }

        # Key bindings - use Habitat default controls
        # Following viewer.py convention:
        # - Arrow keys: turn_left/right (body), look_up/down (sensor)
        # - WASD: body movement
        # - ZX: sensor vertical movement
        self.key_to_action = {
            key.UP: "look_up",        # Pitch up (sensor)
            key.DOWN: "look_down",    # Pitch down (sensor)
            key.LEFT: "turn_left",    # Yaw left (body)
            key.RIGHT: "turn_right",  # Yaw right (body)
            key.A: "move_left",       # Move left (body)
            key.D: "move_right",      # Move right (body)
            key.S: "move_backward",   # Move backward (body)
            key.W: "move_forward",    # Move forward (body)
            key.X: "move_down",       # Move down (sensor)
            key.Z: "move_up",         # Move up (sensor)
        }

        # Setup display font
        self.display_font = text.FontManager().load_and_instantiate("TrueTypeFont")
        relative_path_to_font = "../data/fonts/ProggyClean.ttf"
        self.display_font.open_file(
            os.path.join(os.path.dirname(__file__), relative_path_to_font),
            GaussianViewer.DISPLAY_FONT_SIZE,
        )

        # Glyph cache
        self.glyph_cache = text.GlyphCacheGL(
            mn.PixelFormat.R8_UNORM, mn.Vector2i(256), mn.Vector2i(1)
        )
        self.display_font.fill_glyph_cache(
            self.glyph_cache,
            string.ascii_lowercase
            + string.ascii_uppercase
            + string.digits
            + ":-_+,.! %µ",
        )

        # Magnum Python bindings now expose the GL text renderer as RendererGL.
        self.window_text = text.RendererGL(self.glyph_cache)
        self.window_text.alignment = text.Alignment.TOP_LEFT

        # Text transform
        self.window_text_transform = mn.Matrix3.projection(
            self.framebuffer_size
        ) @ mn.Matrix3.translation(
            mn.Vector2(self.framebuffer_size)
            * mn.Vector2(
                -GaussianViewer.TEXT_DELTA_FROM_CENTER,
                GaussianViewer.TEXT_DELTA_FROM_CENTER,
            )
        )
        self.shader = shaders.VectorGL2D()

        # Set blend function for text
        mn.gl.Renderer.set_blend_equation(
            mn.gl.Renderer.BlendEquation.ADD, mn.gl.Renderer.BlendEquation.ADD
        )

        # Initialize simulator and load scene
        self.sim: habitat_sim.Simulator = None
        self.agent_id: int = 0
        self.gaussian_avatar_manager = None
        self.init_error: Optional[str] = None
        self.init_simulator(window_width, window_height)
        if self.sim is None:
            return
        self._install_nav_collision_debug()

        # Camera control
        self.time_since_last_update = 0.0
        
        # Start timer
        Timer.start()

        self.print_help_text()

    def init_simulator(self, window_width: int, window_height: int) -> None:
        """
        Initialize Habitat-Sim.
        1) Dataset/Scene mode: if dataset and scene provided, build cfg via make_cfg().
        2) Fallback PLY mode: load NONE scene and add 3DGS instance manually.
        """
        # Build sensors common to both modes
        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [self.framebuffer_size[1], self.framebuffer_size[0]]
        color_sensor_spec.position = [0.0, 0.0, 0.0]
        color_sensor_spec.hfov = 90.0

        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [self.framebuffer_size[1], self.framebuffer_size[0]]
        depth_sensor_spec.position = [0.0, 0.0, 0.0]
        depth_sensor_spec.hfov = 90.0
        # Depth output should be single-channel
        depth_sensor_spec.channels = 1

        # Mode 1: dataset + scene
        if self.dataset is not None and self.scene is not None:
            logger.info(f"Using dataset+scene mode: dataset={self.dataset}, scene={self.scene}")
            sim_settings: Dict[str, Any] = dict(default_sim_settings)
            sim_settings["scene_dataset_config_file"] = self.dataset
            sim_settings["scene"] = self.scene
            sim_settings["enable_physics"] = self.enable_physics
            sim_settings["window_width"] = window_width
            sim_settings["window_height"] = window_height
            sim_settings["width"] = self.framebuffer_size[0]
            sim_settings["height"] = self.framebuffer_size[1]
            sim_settings["default_agent"] = 0
            sim_settings["default_agent_navmesh"] = False
            sim_settings["enable_hbao"] = True
            sim_settings["gaussian_auto_play"] = self.time_playing
            if self.user_time_override:
                sim_settings["gaussian_time"] = self.current_time
            cfg = make_cfg(sim_settings)
            # Overwrite sensors on agent 0
            cfg.agents[0].sensor_specifications = [color_sensor_spec, depth_sensor_spec]
            # IMPORTANT: use GS-specific action space so keys map to valid actions
            cfg.agents[0].action_space = self.create_action_space()
            try:
                self.sim = habitat_sim.Simulator(cfg)
            except Exception as e:
                self.init_error = f"Failed to create simulator (dataset mode): {e}"
                logger.error(self.init_error)
                return
        else:
            # Fallback PLY mode
            if not self.gaussian_file:
                self.init_error = "Either provide --dataset + --scene or --input PLY."
                logger.error(self.init_error)
                return
            logger.info(f"Fallback PLY mode: loading {self.gaussian_file}")
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = "NONE"
            sim_cfg.enable_physics = self.enable_physics
            sim_cfg.create_renderer = True
            sim_cfg.gpu_device_id = 0
            sim_cfg.enable_hbao = True
            sim_cfg.gaussian_auto_play = self.time_playing
            if self.user_time_override:
                sim_cfg.gaussian_time = self.current_time

            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = [color_sensor_spec, depth_sensor_spec]
            agent_cfg.action_space = self.create_action_space()
            agent_cfg.height = 1.5
            agent_cfg.radius = 0.1
            cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
            try:
                self.sim = habitat_sim.Simulator(cfg)
            except Exception as e:
                self.init_error = f"Failed to create simulator (PLY mode): {e}"
                logger.error(self.init_error)
                logger.error("Make sure CUDA is enabled and the PLY file is valid")
                return

            # Manually add GS instance for rendering
            try:
                render_helper = habitat_sim.RenderInstanceHelper(self.sim, use_xyzw_orientations=False)
                instance_idx = render_helper.add_instance(
                    asset_filepath=self.gaussian_file,
                    semantic_id=0,
                    scale=mn.Vector3(1.0, 1.0, 1.0)
                )
                positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                orientations = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
                render_helper.set_world_poses(positions, orientations)
                logger.info(f"Gaussian Splatting instance loaded (idx={instance_idx})")
            except Exception as e:
                logger.error(f"Failed to load Gaussian Splatting asset: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Cache agent and sensors
        self.default_agent = self.sim.get_agent(self.agent_id)
        self.color_sensor_wrapper = self.sim.get_sensor("color_sensor")
        self.depth_sensor_wrapper = self.sim.get_sensor("depth_sensor")
        self.render_camera = self.default_agent.scene_node.node_sensor_suite.get("color_sensor")
        self.depth_sensor = self.default_agent.scene_node.node_sensor_suite.get("depth_sensor")
        self.gaussian_avatar_manager = getattr(self.sim, "_gaussian_avatar_manager", None)
        if self.gaussian_avatar_manager is not None:
            logger.info(
                "GaussianAvatar auto-management active (%d avatar instance(s)).",
                len(self.gaussian_avatar_manager.avatars),
            )

        # Set initial agent state
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([0.0, 1.5, 3.0])
        agent_state.rotation = quat_from_angle_axis(0.0, np.array([0, 1, 0]))
        self.default_agent.set_state(agent_state)
        self.previous_mouse_point = mn.Vector2i(
            self.framebuffer_size[0] // 2,
            self.framebuffer_size[1] // 2,
        )

        # Automatically compute NavMesh if not already loaded and scene is not NONE
        if (
            not self.sim.pathfinder.is_loaded
            and self.sim.curr_scene_name.lower() != "none"
        ):
            logger.info("Auto-computing NavMesh on initialization...")
            self.navmesh_config_and_recompute()

        logger.info("Simulator initialized successfully")
        self._sync_time_from_sim()
        self._refresh_gaussian_avatars()
        if self.user_time_override:
            self.set_time(self.current_time)

    def _install_nav_collision_debug(self) -> None:
        if self.sim is None:
            return
        try:
            agent = self.sim.get_agent(self.agent_id)
        except Exception:
            return
        controls = agent.controls
        if self._orig_move_filter_fn is None:
            self._orig_move_filter_fn = controls.move_filter_fn

        def _debug_filter(start_pos, end_pos):
            filter_end = self._orig_move_filter_fn(start_pos, end_pos)
            if self.debug_nav_collision:
                start = mn.Vector3(start_pos)
                end = mn.Vector3(end_pos)
                filt = mn.Vector3(filter_end)
                dist_before = (end - start).dot()
                dist_after = (filt - start).dot()
                collided = (dist_after + 1e-5) < dist_before
                now = time.time()
                if collided or (now - self._last_nav_filter_log) > self._nav_filter_log_interval:
                    logger.info(
                        "NavFilter[%s] nav_loaded=%s moved=%.3f->%.3f collided=%s start=%s end=%s filt=%s",
                        self._last_action_name,
                        bool(self.sim.pathfinder.is_loaded),
                        dist_before,
                        dist_after,
                        collided,
                        np.array(start),
                        np.array(end),
                        np.array(filt),
                    )
                    self._last_nav_filter_log = now
            return filter_end

        controls.move_filter_fn = _debug_filter

    def _sync_time_from_sim(self) -> None:
        """Pull Gaussian timing parameters from the C++ simulator."""
        try:
            self.current_time = float(self.sim.gaussian_time)
            self.time_step = 0.0
            self.time_max = (
                float(self.sim.gaussian_time_max)
                if self.sim.gaussian_time_max > 0.0
                else None
            )
            self.loop_time = bool(self.sim.gaussian_time_loop)
            if not self.user_autoplay_override:
                self.time_playing = bool(self.sim.gaussian_auto_play)
            # Keep simulator autoplay in sync with local play state.
            self._apply_play_state(self.time_playing)
        except Exception as ex:  # pragma: no cover - best-effort fallback
            logger.warning(f"Failed to sync Gaussian timing from simulator: {ex}")

    def _refresh_gaussian_avatars(self) -> None:
        """Force a pose refresh after manual Gaussian time edits."""
        if self.sim is None:
            return

        self.gaussian_avatar_manager = getattr(
            self.sim, "_gaussian_avatar_manager", self.gaussian_avatar_manager
        )
        try:
            if hasattr(self.sim, "_update_gaussian_avatars"):
                self.sim._update_gaussian_avatars()
            elif self.gaussian_avatar_manager is not None:
                self.gaussian_avatar_manager.update(self.sim)
        except Exception as exc:
            logger.warning(f"GaussianAvatar refresh failed: {exc}")

    def _step_size(self) -> float:
        """Return a positive step size for manual scrubbing."""
        if self.time_step > 0.0:
            return self.time_step
        try:
            step = float(self.sim.get_physics_time_step())
            if step > 0.0:
                return step
        except Exception:
            pass
        if Timer.prev_frame_duration > 0.0:
            return Timer.prev_frame_duration
        return 1.0 / 30.0

    def _apply_play_state(self, play: bool) -> None:
        """Update local + simulator autoplay flag."""
        self.time_playing = play
        try:
            self.sim.gaussian_auto_play = bool(play)
        except Exception:
            pass

    def create_action_space(self) -> Dict[str, habitat_sim.agent.ActionSpec]:
        """Create action space for agent movement using Gaussian Splatting controls."""
        MOVE, LOOK = 0.07, 1.5
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec

        # Use Habitat default control actions
        action_list = [
            "move_left",
            "turn_left",
            "move_right",
            "turn_right",
            "move_backward",
            "look_up",
            "move_forward",
            "look_down",
            "move_down",
            "move_up",
        ]

        action_space = {}
        for action in action_list:
            actuation_spec_amt = MOVE if "move" in action else LOOK
            action_spec = make_action_spec(
                action, make_actuation_spec(actuation_spec_amt)
            )
            action_space[action] = action_spec

        return action_space

    def toggle_render_mode(self) -> None:
        """Toggle between RGB and Depth rendering modes."""
        if self.render_mode == RenderMode.RGB:
            self.render_mode = RenderMode.DEPTH
            logger.info("Switched to DEPTH mode")
        else:
            self.render_mode = RenderMode.RGB
            logger.info("Switched to RGB mode")

    def draw_event(self) -> None:
        """Main drawing loop."""
        agent_acts_per_sec = self.fps

        # Clear framebuffer
        mn.gl.default_framebuffer.clear(
            mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH
        )

        # Handle agent actions
        self.time_since_last_update += Timer.prev_frame_duration
        num_agent_actions: int = self.time_since_last_update * agent_acts_per_sec
        self.move_and_look(int(num_agent_actions))

        if self.time_since_last_update >= 1.0 / self.fps:
            # Reset timer
            self.time_since_last_update = math.fmod(
                self.time_since_last_update, 1.0 / self.fps
            )

        # Render based on current mode
        try:
            prev_time = self.current_time
            step_dt = 1.0 / self.fps

            if self.sim is not None and (
                self.time_playing or self.simulating or self.simulate_single_step
            ):
                self.sim.step_world(step_dt)
                self.simulate_single_step = False

            if self.sim is not None and self.time_playing and abs(self.time_rate - 1.0) > 1.0e-6:
                # Apply viewer-side Gaussian time scaling without changing physics dt.
                self.sim.gaussian_time = float(prev_time + step_dt * self.time_rate)
                self._refresh_gaussian_avatars()

            if self.sim is not None:
                self.current_time = float(self.sim.gaussian_time)

            if self.render_mode == RenderMode.RGB:
                self.color_sensor_wrapper.draw_observation()
                self.debug_draw()
                self.render_camera.render_target.blit_rgba_to_default()
            else:
                self.depth_sensor_wrapper.draw_observation()
                depth_obs = self.depth_sensor_wrapper.get_observation()
                if depth_obs is not None:
                    self.visualize_depth(depth_obs)
                else:
                    logger.warning("No depth observation available")

        except Exception as e:
            logger.error(f"Rendering error: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # Draw overlay text
        mn.gl.default_framebuffer.bind()
        self.draw_text()

        # Swap buffers
        self.swap_buffers()
        Timer.next_frame()
        self.redraw()

    def draw_contact_debug(self, debug_line_render: Any) -> None:
        """Draw active contact points and normals."""
        yellow = mn.Color4.yellow()
        red = mn.Color4.red()
        cps = self.sim.get_physics_contact_points()
        debug_line_render.set_line_width(1.5)
        camera_position = self.render_camera.render_camera.node.absolute_translation
        for cp in (contact for contact in cps if contact.is_active):
            debug_line_render.draw_transformed_line(
                cp.position_on_b_in_ws,
                cp.position_on_b_in_ws
                + cp.contact_normal_on_b_in_ws * -cp.contact_distance,
                red,
            )
            debug_line_render.draw_transformed_line(
                cp.position_on_b_in_ws,
                cp.position_on_b_in_ws + cp.contact_normal_on_b_in_ws * 0.1,
                yellow,
            )
            debug_line_render.draw_circle(
                translation=cp.position_on_b_in_ws,
                radius=0.005,
                color=yellow,
                normal=camera_position - cp.position_on_b_in_ws,
            )

    def draw_region_debug(self, debug_line_render: Any) -> None:
        """Draw semantic region wireframes when available."""
        if not hasattr(self.sim, "semantic_scene") or self.sim.semantic_scene is None:
            return
        for region in self.sim.semantic_scene.regions:
            color = self.debug_semantic_colors.get(region.id, mn.Color4.magenta())
            for edge in region.volume_edges:
                debug_line_render.draw_transformed_line(edge[0], edge[1], color)

    def debug_draw(self) -> None:
        """Draw physics, semantic, and avatar debug overlays."""
        if not (
            self.debug_bullet_draw
            or self.contact_debug_draw
            or self.semantic_region_debug_draw
            or self.show_avatar_proxies
        ):
            return

        if self.debug_bullet_draw:
            render_cam = self.render_camera.render_camera
            proj_mat = render_cam.projection_matrix.__matmul__(render_cam.camera_matrix)
            self.sim.physics_debug_draw(proj_mat)

        debug_line_render = self.sim.get_debug_line_render()
        if debug_line_render is None:
            return

        if self.contact_debug_draw:
            self.draw_contact_debug(debug_line_render)

        if self.semantic_region_debug_draw and hasattr(self.sim, "semantic_scene"):
            if len(self.debug_semantic_colors) != len(self.sim.semantic_scene.regions):
                for region in self.sim.semantic_scene.regions:
                    self.debug_semantic_colors[region.id] = mn.Color4(
                        mn.Vector3(np.random.random(3))
                    )
            self.draw_region_debug(debug_line_render)

        if self.show_avatar_proxies:
            self.draw_avatar_proxy_capsules(debug_line_render)

    def draw_avatar_proxy_capsules(self, debug_line_render: Any) -> None:
        """Draw the current precomputed navmesh proxy capsules for each avatar."""
        manager = getattr(self.sim, "_gaussian_avatar_manager", self.gaussian_avatar_manager)
        if manager is None:
            return

        camera_position = self.render_camera.render_camera.node.absolute_translation
        palette = [
            mn.Color4(0.10, 0.85, 0.30, 1.0),
            mn.Color4(1.00, 0.67, 0.10, 1.0),
            mn.Color4(0.20, 0.75, 1.00, 1.0),
            mn.Color4(0.95, 0.35, 0.55, 1.0),
        ]
        debug_line_render.set_line_width(2.0)
        for avatar_idx, avatar in enumerate(manager.avatars):
            capsules = avatar.get_navmesh_capsules()
            if capsules is None or capsules.size == 0:
                continue

            color = palette[avatar_idx % len(palette)]
            for capsule in capsules:
                p0 = mn.Vector3(capsule[:3])
                p1 = mn.Vector3(capsule[3:6])
                radius = float(capsule[6])
                self._draw_wire_capsule(
                    debug_line_render, p0, p1, radius, color, camera_position
                )

    def _draw_wire_capsule(
        self,
        debug_line_render: Any,
        p0: mn.Vector3,
        p1: mn.Vector3,
        radius: float,
        color: mn.Color4,
        camera_position: mn.Vector3,
    ) -> None:
        """Approximate a capsule with axis rings and side rails."""
        axis = p1 - p0
        axis_len = axis.length()
        if radius <= 0.0:
            return
        if axis_len <= 1.0e-5:
            normal = camera_position - p0
            if normal.length() <= 1.0e-5:
                normal = mn.Vector3.y_axis()
            debug_line_render.draw_circle(
                translation=p0,
                radius=radius,
                color=color,
                num_segments=16,
                normal=normal,
            )
            return

        axis_dir = axis / axis_len
        ref = mn.Vector3.y_axis()
        if abs(mn.math.dot(axis_dir, ref)) > 0.95:
            ref = mn.Vector3.x_axis()
        tangent_a = mn.math.cross(axis_dir, ref).normalized()
        tangent_b = mn.math.cross(axis_dir, tangent_a).normalized()
        midpoint = (p0 + p1) * 0.5

        debug_line_render.draw_transformed_line(p0, p1, color)
        for center in (p0, midpoint, p1):
            debug_line_render.draw_circle(
                translation=center,
                radius=radius,
                color=color,
                num_segments=16,
                normal=axis_dir,
            )

        for tangent in (tangent_a, -tangent_a, tangent_b, -tangent_b):
            debug_line_render.draw_transformed_line(
                p0 + tangent * radius,
                p1 + tangent * radius,
                color,
            )
    
    def visualize_depth(self, depth_data: np.ndarray) -> None:
        """
        Visualize depth data with spectral colormap.
        
        Args:
            depth_data: Raw depth values as numpy array
        """
        # print(f"[DEBUG] depth range: {depth_data.min()}, {depth_data.max()}")

        # Ensure depth is 2D float data
        depth_image = np.squeeze(np.asarray(depth_data))
        if depth_image.ndim > 2:
            depth_image = depth_image[..., 0]
        depth_image = depth_image.astype(np.float32, copy=False)

        # Auto-adjust depth range based on actual data with percentile clipping
        valid_depths = depth_image[(depth_image > 0) & np.isfinite(depth_image)]
        if valid_depths.size > 0:
            # Use percentile-based clipping to filter outliers
            depth_min_clip = np.percentile(valid_depths, self.depth_clip_min_percentile)
            depth_max_clip = np.percentile(valid_depths, self.depth_clip_max_percentile)
            self.depth_min = depth_min_clip
            self.depth_max = depth_max_clip
        if self.depth_max <= self.depth_min:
            self.depth_max = self.depth_min + 1e-4
        
        # Create a mask for invalid depth values (zero or negative)
        # Also filter out depths outside the percentile range
        valid_mask = (depth_image > 0) & np.isfinite(depth_image) & (depth_image >= self.depth_min) & (depth_image <= self.depth_max)
        
        # Normalize depth to [0, 1] range
        depth_normalized = np.clip(depth_image, self.depth_min, self.depth_max)
        depth_normalized = (depth_normalized - self.depth_min) / (self.depth_max - self.depth_min)
        
        # Apply spectral colormap (reversed)
        # Spectral_r goes from red (near) to blue (far)
        depth_colored = self.colormap(depth_normalized)
        
        # Set invalid pixels and clipped pixels to black (zero alpha for transparency)
        depth_colored[~valid_mask] = [0.0, 0.0, 0.0, 0.0]
        
        # Convert to RGBA format (0-255)
        depth_rgba = (depth_colored * 255).astype(np.uint8)
        
        # Create or update texture
        if self.depth_texture is None:
            self.depth_texture = mn.gl.Texture2D()
            self.depth_texture.magnification_filter = mn.gl.SamplerFilter.LINEAR
            self.depth_texture.minification_filter = mn.gl.SamplerFilter.LINEAR
            self.depth_texture.wrapping = (
                mn.gl.SamplerWrapping.CLAMP_TO_EDGE
            )
        
        # Upload texture data
        height, width = depth_image.shape[:2]
        self.depth_texture.set_storage(
            1,
            mn.gl.TextureFormat.RGBA8,
            mn.Vector2i(width, height)
        )
        
        # Flip vertically for OpenGL coordinate system
        depth_rgba_flipped = np.flip(depth_rgba, axis=0)
        
        # Ensure the array is contiguous and in the correct format
        depth_rgba_contiguous = np.ascontiguousarray(depth_rgba_flipped, dtype=np.uint8)
        
        self.depth_texture.set_sub_image(
            0,
            mn.Vector2i(0, 0),
            mn.ImageView2D(
                mn.PixelFormat.RGBA8_UNORM,
                mn.Vector2i(width, height),
                depth_rgba_contiguous
            )
        )
        
        # Create mesh for fullscreen quad if not exists
        if self.depth_mesh is None:
            self.create_fullscreen_quad()
        
        # Create shader if not exists
        if self.depth_shader is None:
            self.depth_shader = mn.shaders.FlatGL2D(flags=mn.shaders.FlatGL2D.Flags.TEXTURED)
        
        # Draw the textured quad
        mn.gl.default_framebuffer.clear(mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH)
        
        # Disable depth test for 2D rendering (restore afterwards to not break
        # subsequent sensor renders)
        mn.gl.Renderer.disable(mn.gl.Renderer.Feature.DEPTH_TEST)
        try:
            self.depth_shader.bind_texture(self.depth_texture)
            self.depth_shader.transformation_projection_matrix = mn.Matrix3.projection(
                mn.Vector2(self.framebuffer_size)
            )
            self.depth_shader.color = mn.Color4(1.0, 1.0, 1.0, 1.0)
            self.depth_shader.draw(self.depth_mesh)
        finally:
            mn.gl.Renderer.enable(mn.gl.Renderer.Feature.DEPTH_TEST)
    
    def create_fullscreen_quad(self) -> None:
        """Create a fullscreen quad mesh for displaying the depth texture."""
        # Vertex positions (screen space)
        width = float(self.framebuffer_size[0])
        height = float(self.framebuffer_size[1])
        
        half_width = width * 0.5
        half_height = height * 0.5

        vertices = np.array([
            [-half_width, -half_height],  # Bottom-left
            [half_width, -half_height],   # Bottom-right
            [half_width, half_height],    # Top-right
            [-half_width, half_height],   # Top-left
        ], dtype=np.float32)
        
        # Texture coordinates
        tex_coords = np.array([
            [0.0, 0.0],  # Bottom-left
            [1.0, 0.0],  # Bottom-right
            [1.0, 1.0],  # Top-right
            [0.0, 1.0],  # Top-left
        ], dtype=np.float32)
        
        # Indices for two triangles
        indices = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32)
        
        # Create mesh
        self.depth_mesh = mn.gl.Mesh()
        self.depth_mesh.primitive = mn.MeshPrimitive.TRIANGLES
        self.depth_mesh.count = indices.size
        
        # Create and configure vertex buffer
        vertex_buffer = mn.gl.Buffer()
        vertex_buffer.set_data(vertices, mn.gl.BufferUsage.STATIC_DRAW)
        
        # Create and configure texture coordinate buffer
        tex_coord_buffer = mn.gl.Buffer()
        tex_coord_buffer.set_data(tex_coords, mn.gl.BufferUsage.STATIC_DRAW)
        
        # Create and configure index buffer
        index_buffer = mn.gl.Buffer()
        index_buffer.set_data(indices, mn.gl.BufferUsage.STATIC_DRAW)
        
        # Precompute attribute strides (in bytes)
        vertex_stride = vertices.strides[0]
        tex_coord_stride = tex_coords.strides[0]

        # Set up mesh with buffers
        self.depth_mesh.add_vertex_buffer(
            vertex_buffer, 0, vertex_stride, mn.shaders.FlatGL2D.POSITION
        )
        self.depth_mesh.add_vertex_buffer(
            tex_coord_buffer, 0, tex_coord_stride, mn.shaders.FlatGL2D.TEXTURE_COORDINATES
        )
        self.depth_mesh.set_index_buffer(
            index_buffer, 0, mn.gl.MeshIndexType.UNSIGNED_INT
        )

    def set_time(self, new_time: float) -> None:
        """Update playback time and propagate to simulator."""
        if self.time_max is not None and self.time_max > 0.0:
            span = max(self.time_max, 1e-6)
            if self.loop_time:
                wrapped = math.fmod(new_time, span)
                if wrapped < 0.0:
                    wrapped += span
                new_time = wrapped
            else:
                new_time = float(np.clip(new_time, 0.0, self.time_max))
        self.current_time = new_time
        if self.sim is not None:
            try:
                self.sim.gaussian_time = float(new_time)
                self._refresh_gaussian_avatars()
            except Exception:
                # Older builds without Gaussian time support will silently ignore.
                pass

    def move_and_look(self, repetitions: int) -> None:
        """Process movement and looking actions."""
        if repetitions == 0:
            return

        agent = self.sim.agents[self.agent_id]
        press = self.pressed
        act = self.key_to_action

        action_queue = [act[k] for k, v in press.items() if v]

        for _ in range(int(repetitions)):
            for action in action_queue:
                self._last_action_name = action
                if self.debug_nav_collision:
                    try:
                        self.sim.perform_discrete_collision_detection()
                    except Exception:
                        pass
                agent.act(action)

        if self.mouse_grabber is not None and self.previous_mouse_point is not None:
            self.update_grab_position(self.previous_mouse_point)

    def invert_gravity(self) -> None:
        """Invert gravity for quick physics debugging."""
        gravity = self.sim.get_gravity() * -1
        self.sim.set_gravity(gravity)

    def navmesh_config_and_recompute(self) -> None:
        """Recompute NavMesh with current agent dimensions."""
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_height = self.sim.agents[self.agent_id].agent_config.height
        navmesh_settings.agent_radius = self.sim.agents[self.agent_id].agent_config.radius
        navmesh_settings.include_static_objects = True

        # navmesh_settings.agent_max_slope = 60.0  
        # navmesh_settings.agent_max_climb = 0.5   
        # navmesh_settings.region_min_size = 40.0 
        # navmesh_settings.edge_max_error = 2.0    
        # navmesh_settings.cell_size = 0.15

        try:
            self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
        except AssertionError as e:
            logger.error(f"Underlying error: {e}")
            return
        except Exception as e:
            logger.error(f"NavMesh recompute failed: {e}")
            return
        # Debug: log navmesh status
        logger.info(f"NavMesh recomputed. PathFinder loaded: {self.sim.pathfinder.is_loaded}")
        logger.info(f"NavMesh has island count: {getattr(self.sim.pathfinder, 'num_islands', 'N/A') if hasattr(self.sim.pathfinder, 'num_islands') else 'N/A'}")
        try:
            ext_bounds = self.sim.pathfinder.get_bounds()
            logger.info(f"NavMesh bounds: min={ext_bounds[0]}, max={ext_bounds[1]}")
        except Exception:
            pass

    def key_press_event(self, event: Application.KeyEvent) -> None:
        """Handle key press events."""
        key = event.key
        pressed = Application.Key
        mod = Application.Modifier

        shift_pressed = bool(event.modifiers & mod.SHIFT)
        alt_pressed = bool(event.modifiers & mod.ALT)

        if key == pressed.ESC:
            event.accepted = True
            self.exit_event(Application.ExitEvent)
            return
        elif key == pressed.H:
            self.print_help_text()
        elif key == pressed.J:
            self.semantic_region_debug_draw = not self.semantic_region_debug_draw
            logger.info(
                "Semantic region overlay: %s",
                "ON" if self.semantic_region_debug_draw else "OFF",
            )
        elif key == pressed.TAB:
            self.toggle_render_mode()
        elif key == pressed.M:
            self.cycle_mouse_mode()
            logger.info(f"Mouse interaction mode: {self.mouse_interaction.name}")
        elif key == pressed.SPACE:
            self._apply_play_state(not self.time_playing)
            state = "playing" if self.time_playing else "paused"
            logger.info(f"Gaussian playback: {state} at t={self.current_time:.3f}s")
        elif key == pressed.P:
            if not self.sim.config.sim_cfg.enable_physics:
                logger.warning("Physics was not enabled during setup.")
            else:
                self.simulating = not self.simulating
                logger.info(
                    "Physics simulation: %s",
                    "RUNNING" if self.simulating else "PAUSED",
                )
        elif key == pressed.PERIOD:
            if not self.sim.config.sim_cfg.enable_physics:
                logger.warning("Physics was not enabled during setup.")
            elif self.simulating:
                logger.warning("Physics is already running continuously.")
            else:
                self.simulate_single_step = True
                logger.info("Stepping physics by one frame.")
        elif key == pressed.COMMA:
            self.debug_bullet_draw = not self.debug_bullet_draw
            logger.info(
                "Bullet collision overlay: %s",
                "ON" if self.debug_bullet_draw else "OFF",
            )
        elif key == pressed.C:
            if shift_pressed:
                self.contact_debug_draw = not self.contact_debug_draw
                logger.info(
                    "Contact debug overlay: %s",
                    "ON" if self.contact_debug_draw else "OFF",
                )
            elif alt_pressed:
                self.debug_nav_collision = not self.debug_nav_collision
                logger.info(
                    "Navmesh collision logs: %s",
                    "ON" if self.debug_nav_collision else "OFF",
                )
            else:
                self.sim.perform_discrete_collision_detection()
                self.contact_debug_draw = True
                logger.info("Contact debug overlay refreshed from a collision pass.")
        elif key == pressed.T:
            fixed_base = alt_pressed
            urdf_file_path = ""
            if shift_pressed and self.cached_urdf:
                urdf_file_path = self.cached_urdf
            else:
                urdf_file_path = input("Load URDF: provide a URDF filepath:").strip()

            if not urdf_file_path:
                logger.warning("Load URDF aborted: no input provided.")
            elif not urdf_file_path.endswith((".URDF", ".urdf")):
                logger.warning("Load URDF aborted: input is not a URDF.")
            elif os.path.exists(urdf_file_path):
                self.cached_urdf = urdf_file_path
                aom = self.sim.get_articulated_object_manager()
                ao = aom.add_articulated_object_from_urdf(
                    urdf_file_path,
                    fixed_base,
                    1.0,
                    1.0,
                    True,
                    maintain_link_order=False,
                    intertia_from_urdf=False,
                )
                ao.translation = self.default_agent.scene_node.transformation.transform_point(
                    [0.0, 1.0, -1.5]
                )
                joint_motor_settings = habitat_sim.physics.JointMotorSettings(
                    position_target=0.0,
                    position_gain=1.0,
                    velocity_target=0.0,
                    velocity_gain=1.0,
                    max_impulse=1000.0,
                )
                for motor_id in ao.existing_joint_motor_ids:
                    ao.remove_joint_motor(motor_id)
                ao.create_all_motors(joint_motor_settings)
                logger.info(
                    "Loaded URDF %s%s",
                    urdf_file_path,
                    " with fixed base" if fixed_base else "",
                )
            else:
                logger.warning("Load URDF aborted: file not found.")
        elif key == pressed.V:
            self.invert_gravity()
            logger.info("Gravity inverted.")
        elif key == pressed.N:
            if alt_pressed:
                logger.info("Resampling agent state from the navmesh.")
                if self.sim.pathfinder.is_loaded:
                    new_state = habitat_sim.AgentState()
                    new_state.position = self.sim.pathfinder.get_random_navigable_point()
                    new_state.rotation = quat_from_angle_axis(
                        np.random.uniform(0, 2.0 * np.pi), np.array([0, 1, 0])
                    )
                    self.default_agent.set_state(new_state)
                else:
                    logger.warning("NavMesh is not initialized.")
            elif shift_pressed:
                logger.info("Recomputing NavMesh.")
                self.navmesh_config_and_recompute()
            else:
                if self.sim.pathfinder.is_loaded:
                    self.sim.navmesh_visualization = not self.sim.navmesh_visualization
                    logger.info(
                        "NavMesh visualization: %s",
                        "ON" if self.sim.navmesh_visualization else "OFF",
                    )
                else:
                    logger.warning("NavMesh not loaded. Use SHIFT+N to compute.")
        elif key == pressed.LEFT_BRACKET:
            self._apply_play_state(False)
            step = self._step_size()
            self.set_time(self.current_time - step)
            logger.info(f"Scrub backward to t={self.current_time:.3f}s")
        elif key == pressed.RIGHT_BRACKET:
            self._apply_play_state(False)
            step = self._step_size()
            self.set_time(self.current_time + step)
            logger.info(f"Scrub forward to t={self.current_time:.3f}s")
        elif key == pressed.O:
            self.show_avatar_proxies = not self.show_avatar_proxies
            state = "ON" if self.show_avatar_proxies else "OFF"
            logger.info(f"GaussianAvatar proxy capsule overlay: {state}")

        # Update movement keys
        if key in self.pressed:
            self.pressed[key] = True
        
        event.accepted = True
        self.redraw()

    def key_release_event(self, event: Application.KeyEvent) -> None:
        """Handle key release events."""
        key = event.key

        if key in self.pressed:
            self.pressed[key] = False
        
        event.accepted = True
        self.redraw()

    def pointer_move_event(self, event: Application.PointerMoveEvent) -> None:
        """Handle mouse movement for camera look and object dragging."""
        if (
            event.pointers & Application.Pointer.MOUSE_LEFT
            and self.mouse_interaction == MouseMode.LOOK
        ):
            agent = self.sim.agents[self.agent_id]
            delta = self.get_mouse_position(event.relative_position) / 2
            action = habitat_sim.agent.ObjectControls()
            act_spec = habitat_sim.agent.ActuationSpec

            action(agent.scene_node, "turn_right", act_spec(delta.x))
            sensors = list(self.default_agent.scene_node.subtree_sensors.values())
            [action(s.object, "look_down", act_spec(delta.y), False) for s in sensors]
        elif self.mouse_interaction == MouseMode.GRAB and self.mouse_grabber:
            self.update_grab_position(self.get_mouse_position(event.position))

        self.previous_mouse_point = self.get_mouse_position(event.position)
        self.redraw()
        event.accepted = True

    def pointer_press_event(self, event: Application.PointerEvent) -> None:
        """Create a point-to-point or fixed constraint on clicked physics objects."""
        physics_enabled = self.sim.get_physics_simulation_library()

        if self.mouse_interaction == MouseMode.GRAB and physics_enabled:
            render_camera = self.render_camera.render_camera
            ray = render_camera.unproject(self.get_mouse_position(event.position))
            raycast_results = self.sim.cast_ray(ray=ray)

            if raycast_results.has_hits():
                hit_object, ao_link = -1, -1
                hit_info = raycast_results.hits[0]

                if hit_info.object_id > habitat_sim.stage_id:
                    ro_mngr = self.sim.get_rigid_object_manager()
                    ao_mngr = self.sim.get_articulated_object_manager()
                    ao = ao_mngr.get_object_by_id(hit_info.object_id)
                    ro = ro_mngr.get_object_by_id(hit_info.object_id)

                    if ro:
                        hit_object = hit_info.object_id
                        object_pivot = ro.transformation.inverted().transform_point(
                            hit_info.point
                        )
                        object_frame = ro.rotation.inverted()
                    elif ao:
                        hit_object = hit_info.object_id
                        object_pivot = ao.transformation.inverted().transform_point(
                            hit_info.point
                        )
                        object_frame = ao.rotation.inverted()
                    else:
                        for ao_handle in ao_mngr.get_objects_by_handle_substring():
                            ao = ao_mngr.get_object_by_handle(ao_handle)
                            link_to_obj_ids = ao.link_object_ids
                            if hit_info.object_id in link_to_obj_ids:
                                ao_link = link_to_obj_ids[hit_info.object_id]
                                object_pivot = (
                                    ao.get_link_scene_node(ao_link)
                                    .transformation.inverted()
                                    .transform_point(hit_info.point)
                                )
                                object_frame = ao.get_link_scene_node(
                                    ao_link
                                ).rotation.inverted()
                                hit_object = ao.object_id
                                break

                    if hit_object >= 0:
                        node = self.default_agent.scene_node
                        constraint_settings = physics.RigidConstraintSettings()
                        constraint_settings.object_id_a = hit_object
                        constraint_settings.link_id_a = ao_link
                        constraint_settings.pivot_a = object_pivot
                        constraint_settings.frame_a = (
                            object_frame.to_matrix() @ node.rotation.to_matrix()
                        )
                        constraint_settings.frame_b = node.rotation.to_matrix()
                        constraint_settings.pivot_b = hit_info.point

                        if event.pointer == Application.Pointer.MOUSE_RIGHT:
                            constraint_settings.constraint_type = (
                                physics.RigidConstraintType.Fixed
                            )

                        grip_depth = (
                            hit_info.point - render_camera.node.absolute_translation
                        ).length()
                        self.mouse_grabber = MouseGrabber(
                            constraint_settings,
                            grip_depth,
                            self.sim,
                        )
                    else:
                        logger.warning("Ray hit an object, but no managed wrapper was found.")

        self.previous_mouse_point = self.get_mouse_position(event.position)
        self.redraw()
        event.accepted = True

    def scroll_event(self, event: Application.ScrollEvent) -> None:
        """Zoom cameras in LOOK mode or adjust the grab constraint in GRAB mode."""
        scroll_mod_val = (
            event.offset.y
            if abs(event.offset.y) > abs(event.offset.x)
            else event.offset.x
        )
        if not scroll_mod_val:
            return

        shift_pressed = bool(event.modifiers & Application.Modifier.SHIFT)
        alt_pressed = bool(event.modifiers & Application.Modifier.ALT)
        ctrl_pressed = bool(event.modifiers & Application.Modifier.CTRL)

        if self.mouse_interaction == MouseMode.LOOK:
            mod_val = 1.01 if shift_pressed else 1.1
            mod = mod_val if scroll_mod_val > 0 else 1.0 / mod_val
            self.render_camera.zoom(mod)
            self.depth_sensor.zoom(mod)
        elif self.mouse_interaction == MouseMode.GRAB and self.mouse_grabber:
            mod_val = 0.1 if shift_pressed else 0.01
            scroll_delta = scroll_mod_val * mod_val
            if alt_pressed or ctrl_pressed:
                agent_t = self.default_agent.scene_node.transformation_matrix()
                rotation_axis = agent_t.transform_vector(mn.Vector3(0, 1, 0))
                if alt_pressed and ctrl_pressed:
                    rotation_axis = agent_t.transform_vector(mn.Vector3(0, 0, -1))
                elif ctrl_pressed:
                    rotation_axis = agent_t.transform_vector(mn.Vector3(1, 0, 0))
                self.mouse_grabber.rotate_local_frame_by_global_angle_axis(
                    rotation_axis, mn.Rad(scroll_delta)
                )
            else:
                self.mouse_grabber.grip_depth += scroll_delta
                self.update_grab_position(self.get_mouse_position(event.position))

        self.redraw()
        event.accepted = True

    def pointer_release_event(self, event: Application.PointerEvent) -> None:
        """Release any active mouse grab constraint."""
        del self.mouse_grabber
        self.mouse_grabber = None
        event.accepted = True

    def update_grab_position(self, point: mn.Vector2i) -> None:
        """Update the target transform of the active mouse constraint."""
        if not self.mouse_grabber:
            return

        render_camera = self.render_camera.render_camera
        ray = render_camera.unproject(point)
        rotation = self.default_agent.scene_node.rotation.to_matrix()
        translation = (
            render_camera.node.absolute_translation
            + ray.direction * self.mouse_grabber.grip_depth
        )
        self.mouse_grabber.update_transform(mn.Matrix4.from_(rotation, translation))

    def cycle_mouse_mode(self) -> None:
        """Toggle between camera look and physics object grab modes."""
        if self.mouse_interaction == MouseMode.LOOK:
            self.mouse_interaction = MouseMode.GRAB
        else:
            self.mouse_interaction = MouseMode.LOOK

    def get_mouse_position(self, mouse_event_position: mn.Vector2i) -> mn.Vector2i:
        """Get scaled mouse position."""
        scaling = mn.Vector2i(self.framebuffer_size) / mn.Vector2i(self.window_size)
        return mouse_event_position * scaling

    def draw_text(self) -> None:
        """Draw overlay text information."""
        mn.gl.Renderer.enable(mn.gl.Renderer.Feature.BLENDING)
        mn.gl.Renderer.set_blend_function(
            mn.gl.Renderer.BlendFunction.ONE,
            mn.gl.Renderer.BlendFunction.ONE_MINUS_SOURCE_ALPHA,
        )

        self.shader.bind_vector_texture(self.glyph_cache.texture)
        self.shader.transformation_projection_matrix = self.window_text_transform
        self.shader.color = [1.0, 1.0, 1.0]

        # Get agent position
        agent = self.sim.get_agent(self.agent_id)
        agent_pos = agent.get_state().position
        pos = agent_pos

        # Build text based on current mode
        mode_text = f"Mode: {self.render_mode.value}"
        if self.render_mode == RenderMode.DEPTH:
            mode_info = f"""
Depth Range: [{self.depth_min:.1f}, {self.depth_max:.1f}]m ({self.depth_clip_min_percentile:.0f}-{self.depth_clip_max_percentile:.0f}%)
Colormap: Spectral_r (Red=Near, Blue=Far)"""
        else:
            mode_info = ""

        scene_label = os.path.basename(self.gaussian_file) if self.gaussian_file else (self.scene or "dataset-scene")
        playback_state = "PLAY" if self.time_playing else "PAUSE"
        step = self._step_size()
        if self.time_max is not None:
            loop_state = "loop" if self.loop_time else "clamp"
            time_info = (
                f"Time: {self.current_time:.3f}s / {self.time_max:.3f}s "
                f"({playback_state}, {loop_state}, step={step:.3f}s, rate={self.time_rate:.2f}x)"
            )
        else:
            time_info = (
                f"Time: {self.current_time:.3f}s "
                f"({playback_state}, step={step:.3f}s, rate={self.time_rate:.2f}x)"
            )

        physics_state = "RUN" if self.simulating else "PAUSE"
        if not self.sim.config.sim_cfg.enable_physics:
            physics_state = "OFF"
        mouse_mode = self.mouse_interaction.name

        self.window_text.clear()
        self.window_text.render(
            self.display_font.create_shaper(),
            self.display_font.size,
            f"""
{self.fps:.1f} FPS
Habitat-GS Viewer
{mode_text}
Scene: {scene_label}
{time_info}
Physics: {physics_state} | Mouse: {mouse_mode}
Camera Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]{mode_info}
TAB: mode | M: mouse mode | H: help | ESC: exit
            """,
        )
        self.shader.draw(self.window_text.mesh)

        mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)

    def exit_event(self, event: Application.ExitEvent) -> None:
        """Clean up and exit."""
        if self.sim:
            self.sim.close(destroy=True)
        event.accepted = True
        exit(0)

    def print_help_text(self) -> None:
        """Print help information."""
        logger.info(
            f"""
=====================================================
Habitat-GS Gaussian Viewer
=====================================================
This viewer supports Gaussian scene inspection, avatar capsule debugging,
physics stepping, and viewer-style object manipulation.

Current Mode: {self.render_mode.value}

RGB Mode:
  - Full-color rendering of the Gaussian scene and any mesh objects

Depth Mode:
  - False-color depth visualization:
    * Red/Orange: Near
    * Green/Yellow: Medium distance
    * Blue/Purple: Far
  - Depth Clipping: {self.depth_clip_min_percentile:.1f}%-{self.depth_clip_max_percentile:.1f}% percentile range

Controls:
---------
  WASD:       Move forward/backward/left/right
  ZX:         Move up/down
  Arrow Keys: Turn left/right, look up/down
  M:          Toggle mouse mode (LOOK / GRAB)
  O:          Toggle GaussianAvatar proxy capsule overlay
  SPACE:      Toggle Gaussian playback
  [:          Step Gaussian time backward ({self._step_size():.3f}s)
  ]:          Step Gaussian time forward ({self._step_size():.3f}s)
  P:          Toggle physics simulation on/off
  .:          Step physics by one frame when paused
  ,:          Toggle Bullet collision wireframe overlay
  C:          Run a collision pass and show active contact points
  SHIFT+C:    Toggle the contact debug overlay
  ALT+C:      Toggle navmesh collision debug logs
  N:          Toggle NavMesh visualization
  SHIFT+N:    Recompute NavMesh
  ALT+N:      Sample a new agent pose from the NavMesh
  T:          Load a URDF
  SHIFT+T:    Reload the previously specified URDF
  ALT+T:      Load the URDF with a fixed base
  V:          Invert gravity
  J:          Toggle semantic region overlay
  Mouse Left: Rotate camera in LOOK mode
  Mouse Left/Right:
              Pick and drag physics objects in GRAB mode
  Scroll:     Zoom cameras in LOOK mode
  Scroll:     Move the grabbed object in GRAB mode
  ALT/CTRL+Scroll:
              Rotate the grabbed constraint frame in GRAB mode

Commands:
---------
  TAB:        Switch between RGB and Depth modes
  H:          Show this help
  ESC:        Exit
=====================================================
"""
        )


class MouseGrabber:
    """Helper around simulator rigid constraints for interactive object dragging."""

    def __init__(
        self,
        settings: physics.RigidConstraintSettings,
        grip_depth: float,
        sim: "habitat_sim.simulator.Simulator",
    ) -> None:
        self.settings = settings
        self.simulator = sim
        self.grip_depth = grip_depth
        self.constraint_id = sim.create_rigid_constraint(settings)

    def __del__(self) -> None:
        if hasattr(self, "constraint_id"):
            self.remove_constraint()

    def remove_constraint(self) -> None:
        """Remove the managed rigid constraint."""
        try:
            self.simulator.remove_rigid_constraint(self.constraint_id)
        except Exception:
            pass

    def update_transform(self, transform: mn.Matrix4) -> None:
        self.settings.frame_b = transform.rotation()
        self.settings.pivot_b = transform.translation
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)

    def rotate_local_frame_by_global_angle_axis(
        self, axis: mn.Vector3, angle: mn.Rad
    ) -> None:
        """Rotate the local constraint frame around a world-space axis."""
        rom = self.simulator.get_rigid_object_manager()
        aom = self.simulator.get_articulated_object_manager()
        if rom.get_library_has_id(self.settings.object_id_a):
            object_transform = rom.get_object_by_id(
                self.settings.object_id_a
            ).transformation
        else:
            object_transform = (
                aom.get_object_by_id(self.settings.object_id_a)
                .get_link_scene_node(self.settings.link_id_a)
                .transformation
            )
        local_axis = object_transform.inverted().transform_vector(axis)
        rotation = mn.Matrix4.rotation(angle, local_axis.normalized())
        self.settings.frame_a = rotation.rotation().__matmul__(self.settings.frame_a)
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)


class Timer:
    """Timer for frame timing."""

    start_time = 0.0
    prev_frame_time = 0.0
    prev_frame_duration = 0.0
    running = False

    @staticmethod
    def start() -> None:
        Timer.running = True
        Timer.start_time = time.time()
        Timer.prev_frame_time = Timer.start_time
        Timer.prev_frame_duration = 0.0

    @staticmethod
    def stop() -> None:
        Timer.running = False
        Timer.start_time = 0.0
        Timer.prev_frame_time = 0.0
        Timer.prev_frame_duration = 0.0

    @staticmethod
    def next_frame() -> None:
        if not Timer.running:
            return
        Timer.prev_frame_duration = time.time() - Timer.prev_frame_time
        Timer.prev_frame_time = time.time()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Habitat-GS viewer with RGB/depth rendering, "
            "avatar capsule debugging, and viewer-style physics manipulation"
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to 3DGS PLY file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Path to scene_dataset_config.json (enables stage-based loading)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=False,
        help="Scene identifier or scene instance file from dataset",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Window width (default: 800)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Window height (default: 600)",
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=0.1,
        help="Minimum depth value for colormap normalization (default: 0.1)",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=10.0,
        help="Maximum depth value for colormap normalization (default: 10.0)",
    )
    parser.add_argument(
        "--depth-clip-min",
        type=float,
        default=1.0,
        help="Minimum percentile for depth clipping (default: 1.0, range: 0-100)",
    )
    parser.add_argument(
        "--depth-clip-max",
        type=float,
        default=99.0,
        help="Maximum percentile for depth clipping (default: 99.0, range: 0-100)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Initial Gaussian time cursor (seconds)",
    )
    parser.add_argument(
        "--time-rate",
        type=float,
        default=1.0,
        help="Playback rate in seconds/second when running (default: 1.0)",
    )
    parser.add_argument(
        "--playback",
        action="store_true",
        default=None,
        help="Start with Gaussian playback enabled",
    )
    parser.add_argument(
        "--enable-physics",
        action="store_true",
        help="Enable physics and NavMesh (requires dataset+scene mode for stage)",
    )

    args = parser.parse_args()

    # Validate inputs (dataset+scene preferred; otherwise require PLY)
    if args.dataset and args.scene:
        if not os.path.exists(args.dataset):
            logger.error(f"Dataset config not found: {args.dataset}")
            return 1
    else:
        if not args.input:
            logger.error("Provide either --dataset + --scene, or --input PLY.")
            return 1
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return 1

    # Check CUDA support
    if not habitat_sim.cuda_enabled:
        logger.error("CUDA support is not enabled. Cannot render Gaussian Splatting.")
        logger.error("Please build habitat-sim with CUDA support.")
        return 1

    logger.info("Starting Habitat-GS Viewer")
    if args.dataset and args.scene:
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Scene: {args.scene}")
    else:
        logger.info(f"Input file: {args.input}")
    logger.info(f"Depth range: [{args.depth_min}, {args.depth_max}]")
    logger.info(f"Depth clipping percentiles: [{args.depth_clip_min}%, {args.depth_clip_max}%]")
    logger.info("Press H for controls, TAB for RGB/depth mode switching")

    # Create and run viewer
    viewer = GaussianViewer(
        args.input, 
        args.width, 
        args.height,
        depth_clip_min_percentile=args.depth_clip_min,
        depth_clip_max_percentile=args.depth_clip_max,
        dataset=args.dataset,
        scene=args.scene,
        start_time=args.time,
        time_rate=args.time_rate,
        autoplay=args.playback,
        enable_physics=bool(args.enable_physics),
    )
    if viewer.sim is None:
        return 1
    viewer.depth_min = args.depth_min
    viewer.depth_max = args.depth_max
    viewer.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())
