from __future__ import annotations

import os
import re
import struct
import subprocess
import tempfile
import zlib
from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np

import habitat_sim

from .types import (
    _DEFAULT_DEPTH_VIS_MAX,
    _DEFAULT_MAX_OBSERVATION_ELEMENTS,
    _DEFAULT_VIDEO_FPS,
    _DEFAULT_VISUAL_OUTPUT_DIR,
    _Session,
    HabitatAdapterError,
)


class HabitatAdapterVisualMediaMixin:
    """Observation serialization, visuals export, and replay/video tooling."""

    @staticmethod
    def _session_output_dir(output_dir: str, session_id: str) -> str:
        """Return session-scoped output directory, creating it if needed.

        If *output_dir* already ends with the session_id component (e.g.
        because the caller is nav_agent running inside a loop subdir),
        return it unchanged to avoid doubling the session path segment.
        """
        # Check if the last or second-to-last path component is the session_id
        tail = os.path.basename(os.path.normpath(output_dir))
        parent_tail = os.path.basename(os.path.dirname(os.path.normpath(output_dir)))
        if tail == session_id or parent_tail == session_id:
            os.makedirs(output_dir, exist_ok=True)
            return output_dir
        session_dir = os.path.join(output_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def _get_observation(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        refresh = self._coerce_bool(
            payload.get("refresh", False), field_name="payload.refresh"
        )
        include_data = self._coerce_bool(
            payload.get("include_observation_data", False),
            field_name="payload.include_observation_data",
        )
        max_elements = self._coerce_int(
            payload.get(
                "max_observation_elements", _DEFAULT_MAX_OBSERVATION_ELEMENTS
            ),
            field_name="payload.max_observation_elements",
        )

        if refresh or session.last_sensor_obs is None:
            session.last_sensor_obs = self._capture_sensor_observations(session)

        return {
            "session_id": session.session_id,
            "step_count": session.step_count,
            "observation": self._serialize_observation(
                session.last_sensor_obs,
                include_data=include_data,
                max_elements=max_elements,
            ),
        }

    def _get_visuals(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        refresh = self._coerce_bool(
            payload.get("refresh", False), field_name="payload.refresh"
        )
        include_metrics = self._coerce_bool(
            payload.get("include_metrics", False),
            field_name="payload.include_metrics",
        )
        output_dir = payload.get("output_dir", _DEFAULT_VISUAL_OUTPUT_DIR)
        if not isinstance(output_dir, str) or not output_dir:
            raise HabitatAdapterError(
                'Field "payload.output_dir" must be a non-empty str'
            )
        depth_max = self._coerce_float(
            payload.get("depth_max", _DEFAULT_DEPTH_VIS_MAX),
            field_name="payload.depth_max",
        )
        if depth_max <= 0:
            raise HabitatAdapterError('Field "payload.depth_max" must be > 0')

        sensors_payload = payload.get("sensors")
        sensors: Optional[list[str]]
        if sensors_payload is None:
            sensors = None
        elif isinstance(sensors_payload, list) and all(
            isinstance(item, str) for item in sensors_payload
        ):
            sensors = sensors_payload
        else:
            raise HabitatAdapterError(
                'Field "payload.sensors" must be list[str] when provided'
            )

        if refresh or session.last_sensor_obs is None:
            session.last_sensor_obs = self._capture_sensor_observations(session)

        session_dir = self._session_output_dir(output_dir, session.session_id)
        session.capture_counter += 1
        visuals = self._export_visuals(
            observation=session.last_sensor_obs,
            output_dir=session_dir,
            sensors=sensors,
            depth_max=depth_max,
            session_id=session.session_id,
            step_count=session.step_count,
            capture_seq=session.capture_counter,
        )
        response = {
            "session_id": session.session_id,
            "step_count": session.step_count,
            "output_dir": session_dir,
            "visuals": visuals,
        }
        if include_metrics:
            response["metrics"] = self._build_metrics(session)
        return response

    _PANORAMA_DIRECTIONS = ("front", "right", "back", "left")

    def _get_panorama(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        include_depth = self._coerce_bool(
            payload.get("include_depth_analysis", False),
            field_name="payload.include_depth_analysis",
        )
        clearance = float(payload.get("clearance_threshold", 0.5))
        output_dir = payload.get("output_dir", _DEFAULT_VISUAL_OUTPUT_DIR)
        if not isinstance(output_dir, str) or not output_dir:
            raise HabitatAdapterError(
                'Field "payload.output_dir" must be a non-empty str'
            )
        depth_max = self._coerce_float(
            payload.get("depth_max", _DEFAULT_DEPTH_VIS_MAX),
            field_name="payload.depth_max",
        )
        if include_depth:
            # Verify depth sensor is available before starting panorama capture
            test_obs = self._capture_sensor_observations(session)
            if "depth_sensor" not in test_obs:
                raise HabitatAdapterError(
                    "get_panorama with include_depth_analysis=true requires "
                    "depth_sensor — enable it with hab init --depth"
                )

        agent = session.simulator.get_agent(session.agent_id)
        saved_state = agent.get_state()

        images = []
        depth_analysis = [] if include_depth else None

        # Use ONE capture_seq for all 4 directions so dashboard can find them
        # at the same step number (all share pano_seq in filename).
        session.capture_counter += 1
        pano_seq = session.capture_counter

        session_dir = self._session_output_dir(output_dir, session.session_id)
        for i, direction in enumerate(self._PANORAMA_DIRECTIONS):
            obs = self._capture_sensor_observations(session)
            heading = self._heading_degrees(session)

            visuals = self._export_visuals(
                observation=obs,
                output_dir=session_dir,
                sensors=["color_sensor"],
                depth_max=depth_max,
                session_id=session.session_id,
                step_count=session.step_count,
                capture_seq=pano_seq,
                filename_prefix=f"pano_{direction}",
            )
            color_info = visuals.get("color_sensor", {})
            images.append({
                "direction": direction,
                "heading_deg": round(heading, 1),
                "path": color_info.get("path"),
            })

            if include_depth and "depth_sensor" in obs:
                depth_arr = np.asarray(obs["depth_sensor"], dtype=np.float32)
                if depth_arr.ndim == 3:
                    depth_arr = depth_arr[:, :, 0]
                analysis = self._analyze_depth_array(depth_arr, clearance)
                depth_analysis.append({
                    "direction": direction,
                    "heading_deg": round(heading, 1),
                    "min_dist": analysis.get("front_center", {}).get("min_dist"),
                    "mean_dist": analysis.get("front_center", {}).get("mean_dist"),
                    "clear": analysis.get("front_center", {}).get("clear", False),
                })

            # Turn 90 degrees right for next direction (9 atomic steps × 10°)
            if i < 3:
                for _ in range(9):
                    session.simulator.step("turn_right")

        # Restore original state (undo all rotations)
        agent.set_state(saved_state, infer_sensor_states=True)
        session.last_sensor_obs = None  # invalidate cached observation

        result: Dict[str, Any] = {
            "session_id": session.session_id,
            "images": images,
        }
        if depth_analysis is not None:
            result["depth_analysis"] = depth_analysis
        return result

    def _export_video_trace(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        output_dir = payload.get("output_dir", _DEFAULT_VISUAL_OUTPUT_DIR)
        if not isinstance(output_dir, str) or not output_dir:
            raise HabitatAdapterError(
                'Field "payload.output_dir" must be a non-empty str'
            )

        sensor = payload.get("sensor", "color_sensor")
        if not isinstance(sensor, str) or not sensor:
            raise HabitatAdapterError(
                'Field "payload.sensor" must be a non-empty str'
            )

        fps = self._coerce_float(
            payload.get("fps", _DEFAULT_VIDEO_FPS),
            field_name="payload.fps",
        )
        if fps <= 0:
            raise HabitatAdapterError('Field "payload.fps" must be > 0')

        step_start_raw = payload.get("step_start")
        step_end_raw = payload.get("step_end")
        step_start = (
            self._coerce_int(step_start_raw, "payload.step_start")
            if step_start_raw is not None
            else None
        )
        step_end = (
            self._coerce_int(step_end_raw, "payload.step_end")
            if step_end_raw is not None
            else None
        )
        if step_start is not None and step_end is not None and step_end < step_start:
            raise HabitatAdapterError(
                'Field "payload.step_end" must be >= "payload.step_start"'
            )

        include_metrics = self._coerce_bool(
            payload.get("include_metrics", True),
            field_name="payload.include_metrics",
        )
        include_publish_hints = self._coerce_bool(
            payload.get("include_publish_hints", True),
            field_name="payload.include_publish_hints",
        )
        frame_records = self._find_visual_frame_records(
            output_dir=output_dir,
            session_id=session.session_id,
            sensor=sensor,
            step_start=step_start,
            step_end=step_end,
        )
        if not frame_records:
            raise HabitatAdapterError(
                "No trace frames found for "
                f'session="{session.session_id}" sensor="{sensor}" '
                f'in "{output_dir}"'
            )

        first_step = frame_records[0][0]
        last_step = frame_records[-1][0]
        frame_paths = [path for _, path in frame_records]
        safe_sensor = self._safe_sensor_name(sensor)
        # Place video in the same directory where frames were found
        video_dir = os.path.dirname(frame_paths[0])
        video_path = os.path.join(
            video_dir,
            f"steps{first_step:06d}-{last_step:06d}_{safe_sensor}.mp4",
        )
        self._encode_video_trace(frame_paths=frame_paths, video_path=video_path, fps=fps)

        video_item: Dict[str, Any] = {
            "ok": True,
            "sensor": sensor,
            "path": video_path,
            "bytes": os.path.getsize(video_path),
            "mime_type": "video/mp4",
            "frame_count": len(frame_paths),
            "fps": fps,
            "duration_s": round(len(frame_paths) / fps, 3),
            "step_start": first_step,
            "step_end": last_step,
        }
        response: Dict[str, Any] = {
            "session_id": session.session_id,
            "output_dir": output_dir,
            "video": video_item,
        }
        if include_metrics:
            metrics = self._build_metrics(session)
            response["metrics"] = metrics
        else:
            metrics = None
        if include_publish_hints:
            response["publish_hints"] = self._build_video_publish_hints(
                session=session,
                video_item=video_item,
                metrics=metrics,
            )
        return response

    def _get_metrics(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        del payload
        session = self._require_session(session_id)
        return self._build_metrics(session)

    def _build_metrics(self, session: _Session) -> Dict[str, Any]:
        agent = session.simulator.get_agent(session.agent_id)
        state = agent.get_state()
        step_time_s = getattr(session.simulator, "_previous_step_time", None)

        metrics: Dict[str, Any] = {
            "session_id": session.session_id,
            "scene": session.scene,
            "step_count": session.step_count,
            "last_action": session.last_action,
            "available_actions": self._get_available_actions(session),
            "trajectory_length": len(session.trajectory),
            "collision_count": len(session.collision_points),
            "last_collision": session.last_collision,
            "state_summary": self._build_state_summary(session),
            "step_time_s": float(step_time_s) if step_time_s is not None else None,
        }
        if not session.mapless:
            metrics["agent_state"] = {
                "position": self._to_numeric_list(state.position),
                "rotation": self._to_numeric_list(state.rotation),
            }
            metrics["current_goal"] = session.last_goal
        else:
            metrics["agent_state"] = {
                "heading_deg": round(self._heading_degrees(session), 3),
            }
        return metrics

    def _build_publish_hints(
        self,
        session: _Session,
        action: str,
        collided: bool,
        visuals: Mapping[str, Mapping[str, Any]],
        metrics: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        media_items: list[Dict[str, Any]] = []
        for sensor_name, item in visuals.items():
            if not item.get("ok", False):
                continue
            path = item.get("path")
            if not isinstance(path, str):
                continue
            media_items.append(
                {
                    "sensor": sensor_name,
                    "path": path,
                    "mime_type": item.get("mime_type", "image/png"),
                }
            )

        if metrics is not None:
            position = metrics.get("agent_state", {}).get("position")
            step_time_s = metrics.get("step_time_s")
        else:
            position = None
            step_time_s = None

        summary_parts = [
            f"scene={session.scene}",
            f"step={session.step_count}",
            f"action={action}",
            f"collided={collided}",
        ]
        if position is not None:
            summary_parts.append(f"position={position}")
        if step_time_s is not None:
            summary_parts.append(f"step_time_s={step_time_s:.6f}")
        summary_text = " | ".join(summary_parts)
        for item in media_items:
            item["label"] = f"{item['sensor']} frame"

        return {
            "summary_text": summary_text,
            "media_items": media_items,
            "agent_send_actions": self._build_agent_send_actions(
                summary_text=summary_text,
                media_items=media_items,
            ),
        }

    def _build_video_publish_hints(
        self,
        session: _Session,
        video_item: MutableMapping[str, Any],
        metrics: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        position = None
        if metrics is not None:
            position = metrics.get("agent_state", {}).get("position")

        summary_parts = [
            f"scene={session.scene}",
            ("trace_steps=" f"{video_item['step_start']}-{video_item['step_end']}"),
            f"frames={video_item['frame_count']}",
            f"fps={video_item['fps']}",
            f"duration_s={video_item['duration_s']:.3f}",
        ]
        if position is not None:
            summary_parts.append(f"position={position}")
        summary_text = " | ".join(summary_parts)

        video_item["label"] = f"{video_item['sensor']} trace video"
        return {
            "summary_text": summary_text,
            "media_items": [dict(video_item)],
            "agent_send_actions": self._build_agent_send_actions(
                summary_text=summary_text,
                media_items=[video_item],
            ),
        }

    @staticmethod
    def _serialize_observation(
        observation: Optional[Mapping[str, Any]],
        include_data: bool,
        max_elements: int,
    ) -> Dict[str, Any]:
        if observation is None:
            return {}

        serialized: Dict[str, Any] = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                serialized[key] = HabitatAdapterVisualMediaMixin._serialize_array(
                    value, include_data=include_data, max_elements=max_elements
                )
            elif hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(
                value, "numpy"
            ):
                np_value = value.detach().cpu().numpy()
                serialized[key] = HabitatAdapterVisualMediaMixin._serialize_array(
                    np_value, include_data=include_data, max_elements=max_elements
                )
            elif isinstance(value, (bool, int, float, str)) or value is None:
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized

    @staticmethod
    def _serialize_array(
        array: np.ndarray, include_data: bool, max_elements: int
    ) -> Dict[str, Any]:
        np_array = np.asarray(array)
        result: Dict[str, Any] = {
            "dtype": str(np_array.dtype),
            "shape": list(np_array.shape),
            "num_elements": int(np_array.size),
        }

        if np_array.size > 0:
            try:
                result["min"] = float(np_array.min())
                result["max"] = float(np_array.max())
            except (TypeError, ValueError):
                pass

        if include_data:
            flat = np_array.reshape(-1)
            if flat.size > max_elements:
                result["data"] = flat[:max_elements].tolist()
                result["truncated"] = True
            else:
                result["data"] = np_array.tolist()
                result["truncated"] = False
        return result

    @staticmethod
    def _to_numeric_list(value: Any) -> Any:
        try:
            import quaternion  # type: ignore

            if isinstance(value, quaternion.quaternion):
                return quaternion.as_float_array(value).tolist()
        except ImportError:
            pass

        if isinstance(value, np.ndarray):
            return value.tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        if hasattr(value, "__iter__"):
            try:
                return list(value)
            except TypeError:
                pass
        return value

    @staticmethod
    def _to_numpy_array(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if (
            hasattr(value, "detach")
            and hasattr(value, "cpu")
            and hasattr(value, "numpy")
        ):
            return value.detach().cpu().numpy()
        raise HabitatAdapterError("Observation is not a numpy-compatible array")

    @staticmethod
    def _to_uint8_color(array: np.ndarray) -> np.ndarray:
        np_array = np.asarray(array)
        if np_array.ndim != 3 or np_array.shape[2] not in (3, 4):
            raise HabitatAdapterError("Color visualization expects shape [H, W, 3|4]")

        if np.issubdtype(np_array.dtype, np.floating):
            np_array = np.nan_to_num(np_array, nan=0.0, posinf=1.0, neginf=0.0)
            max_val = float(np.max(np_array)) if np_array.size else 1.0
            if max_val <= 1.0:
                np_array = np_array * 255.0
            np_array = np.clip(np_array, 0.0, 255.0).astype(np.uint8)
        elif np_array.dtype != np.uint8:
            np_array = np.clip(np_array, 0, 255).astype(np.uint8)
        return np_array

    @staticmethod
    def _to_uint8_depth(array: np.ndarray, depth_max: float) -> np.ndarray:
        depth = np.asarray(array, dtype=np.float32)
        if depth.ndim != 2:
            raise HabitatAdapterError("Depth visualization expects shape [H, W]")
        depth = np.nan_to_num(depth, nan=depth_max, posinf=depth_max, neginf=0.0)
        depth = np.clip(depth, 0.0, depth_max)
        normalized = 1.0 - (depth / depth_max)
        return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _encode_png_bytes(image: np.ndarray) -> bytes:
        image_array = np.asarray(image)
        if image_array.dtype != np.uint8:
            raise HabitatAdapterError("PNG encoder expects uint8 arrays")

        if image_array.ndim == 2:
            height, width = image_array.shape
            color_type = 0
            raw_rows = [image_array[row].tobytes() for row in range(height)]
        elif image_array.ndim == 3 and image_array.shape[2] in (3, 4):
            height, width, channels = image_array.shape
            color_type = 2 if channels == 3 else 6
            raw_rows = [image_array[row].tobytes() for row in range(height)]
        else:
            raise HabitatAdapterError(
                "PNG encoder expects [H, W] or [H, W, 3|4] uint8 arrays"
            )

        scanlines = b"".join(b"\x00" + row_bytes for row_bytes in raw_rows)
        compressed = zlib.compress(scanlines, level=6)

        def chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
            length = struct.pack(">I", len(chunk_data))
            crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
            return length + chunk_type + chunk_data + struct.pack(">I", crc)

        ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
        png_signature = b"\x89PNG\r\n\x1a\n"
        return (
            png_signature
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", compressed)
            + chunk(b"IEND", b"")
        )

    @staticmethod
    def _safe_sensor_name(sensor_name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", sensor_name).strip("_") or "sensor"

    def _parse_visual_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        output_dir = payload.get("output_dir", _DEFAULT_VISUAL_OUTPUT_DIR)
        if not isinstance(output_dir, str) or not output_dir:
            raise HabitatAdapterError(
                'Field "payload.output_dir" must be a non-empty str'
            )
        depth_max = self._coerce_float(
            payload.get("depth_max", _DEFAULT_DEPTH_VIS_MAX),
            field_name="payload.depth_max",
        )
        if depth_max <= 0:
            raise HabitatAdapterError('Field "payload.depth_max" must be > 0')

        sensors_payload = payload.get("sensors")
        if sensors_payload is None:
            sensors = None
        elif isinstance(sensors_payload, list) and all(
            isinstance(item, str) for item in sensors_payload
        ):
            sensors = sensors_payload
        else:
            raise HabitatAdapterError(
                'Field "payload.sensors" must be list[str] when provided'
            )

        return {
            "output_dir": output_dir,
            "depth_max": depth_max,
            "sensors": sensors,
        }

    @staticmethod
    def _build_agent_send_actions(
        summary_text: str, media_items: list[Mapping[str, Any]]
    ) -> list[Dict[str, Any]]:
        agent_send_actions: list[Dict[str, Any]] = [
            {"action": "send", "message": summary_text}
        ]
        for item in media_items:
            media_path = item.get("path")
            if not isinstance(media_path, str):
                continue
            label = item.get("label")
            if not isinstance(label, str) or not label:
                label = "media"
            agent_send_actions.append(
                {
                    "action": "send",
                    "media": media_path,
                    "message": label,
                }
            )
        return agent_send_actions

    @staticmethod
    def _to_rgb_topdown(raw_map: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_map)
        if raw.ndim != 2:
            raise HabitatAdapterError("Topdown map must be a 2D array")
        binary = np.where(raw > 0, 255, 24).astype(np.uint8)
        return np.stack([binary, binary, binary], axis=-1)

    @staticmethod
    def _topdown_xy(
        pathfinder: Any, point: Any, meters_per_pixel: float
    ) -> tuple[int, int]:
        bounds = pathfinder.get_bounds()
        px = int(round((float(point[0]) - float(bounds[0][0])) / meters_per_pixel))
        py = int(round((float(point[2]) - float(bounds[0][2])) / meters_per_pixel))
        return px, py

    @staticmethod
    def _draw_circle(
        image: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]
    ) -> None:
        height, width = image.shape[:2]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                x = cx + dx
                y = cy + dy
                if 0 <= x < width and 0 <= y < height:
                    image[y, x, :3] = color

    @staticmethod
    def _draw_line(
        image: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        steps = max(abs(dx), abs(dy), 1)
        for step in range(steps + 1):
            t = step / steps
            x = int(round(start[0] + dx * t))
            y = int(round(start[1] + dy * t))
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                image[y, x, :3] = color

    def _draw_topdown_marker(
        self,
        image: np.ndarray,
        pathfinder: Any,
        point: Any,
        meters_per_pixel: float,
        color: tuple[int, int, int],
        radius: int = 3,
    ) -> None:
        px, py = self._topdown_xy(pathfinder, point, meters_per_pixel)
        self._draw_circle(image, px, py, radius=radius, color=color)

    def _draw_topdown_path(
        self,
        image: np.ndarray,
        pathfinder: Any,
        points: list[list[float]],
        meters_per_pixel: float,
        color: tuple[int, int, int] = (255, 80, 80),
    ) -> None:
        if len(points) < 2:
            return
        topdown_points = [
            self._topdown_xy(pathfinder, point, meters_per_pixel) for point in points
        ]
        for start, end in zip(topdown_points[:-1], topdown_points[1:]):
            self._draw_line(image, start, end, color)

    @staticmethod
    def _fill_triangle(
        image: np.ndarray,
        a: tuple[int, int],
        b: tuple[int, int],
        c: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        min_x = max(0, min(a[0], b[0], c[0]))
        max_x = min(image.shape[1] - 1, max(a[0], b[0], c[0]))
        min_y = max(0, min(a[1], b[1], c[1]))
        max_y = min(image.shape[0] - 1, max(a[1], b[1], c[1]))

        def _edge(p0: tuple[int, int], p1: tuple[int, int], p2: tuple[int, int]) -> int:
            return (
                (p2[0] - p0[0]) * (p1[1] - p0[1])
                - (p2[1] - p0[1]) * (p1[0] - p0[0])
            )

        area = _edge(a, b, c)
        if area == 0:
            return

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = (x, y)
                w0 = _edge(b, c, p)
                w1 = _edge(c, a, p)
                w2 = _edge(a, b, p)
                if area > 0:
                    inside = w0 >= 0 and w1 >= 0 and w2 >= 0
                else:
                    inside = w0 <= 0 and w1 <= 0 and w2 <= 0
                if inside:
                    image[y, x, :3] = color

    def _draw_topdown_arrow(
        self,
        image: np.ndarray,
        pathfinder: Any,
        point: Any,
        direction: np.ndarray,
        meters_per_pixel: float,
        color: tuple[int, int, int],
    ) -> None:
        start = self._topdown_xy(pathfinder, point, meters_per_pixel)
        direction = np.asarray(direction, dtype=np.float32)
        tip_point = np.asarray(point, dtype=np.float32) + direction * 0.5
        tip = self._topdown_xy(pathfinder, tip_point, meters_per_pixel)
        screen_vec = np.array(
            [float(tip[0] - start[0]), float(tip[1] - start[1])], dtype=np.float32
        )
        norm = float(np.linalg.norm(screen_vec))
        if norm < 1.0:
            self._draw_circle(image, tip[0], tip[1], radius=2, color=color)
            return

        unit = screen_vec / norm
        perp = np.array([-unit[1], unit[0]], dtype=np.float32)
        head_len = min(10.0, max(6.0, norm * 0.35))
        head_half_width = max(3.0, head_len * 0.45)
        base_center = np.array(tip, dtype=np.float32) - unit * head_len
        left_tip = tuple(np.rint(base_center + perp * head_half_width).astype(int))
        right_tip = tuple(np.rint(base_center - perp * head_half_width).astype(int))
        shaft_end = tuple(np.rint(base_center).astype(int))

        self._draw_line(image, start, shaft_end, color)
        self._fill_triangle(image, tip, left_tip, right_tip, color)

    def _write_png_image(
        self,
        image: np.ndarray,
        output_dir: str,
        name: str,
        session_id: str,
        step_count: int,
        capture_seq: Optional[int] = None,
        filename_prefix: str = "",
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        png_bytes = self._encode_png_bytes(image)
        safe_name = self._safe_sensor_name(name)
        # Use capture_seq (monotonic) to avoid overwriting when look is
        # called multiple times at the same step_count.
        seq = capture_seq if capture_seq is not None else step_count
        prefix_part = f"{filename_prefix}_" if filename_prefix else ""
        file_path = os.path.join(
            output_dir, f"{prefix_part}step{seq:06d}_{safe_name}.png"
        )
        with open(file_path, "wb") as file_handle:
            file_handle.write(png_bytes)

        height, width = image.shape[:2]
        item: Dict[str, Any] = {
            "ok": True,
            "sensor": name,
            "mode": "RGB" if image.ndim == 3 else "L",
            "width": int(width),
            "height": int(height),
            "mime_type": "image/png",
            "path": file_path,
            "bytes": len(png_bytes),
        }
        return item

    def _find_visual_frame_records(
        self,
        output_dir: str,
        session_id: str,
        sensor: str,
        step_start: Optional[int],
        step_end: Optional[int],
    ) -> list[tuple[int, str]]:
        """Find frame files for video export.

        The sequence number in filenames is capture_seq (monotonic per
        session), which may diverge from simulator step_count when look
        is called multiple times at the same step.  step_start/step_end
        filter on capture_seq, not simulator step_count.

        Searches in the session subdirectory (output_dir/session_id/) first,
        falling back to output_dir for backward compatibility.
        """
        session_dir = os.path.join(output_dir, session_id)
        # Prefer session subdir; fall back to flat dir for old layouts
        search_dir = session_dir if os.path.isdir(session_dir) else output_dir
        if not os.path.isdir(search_dir):
            raise HabitatAdapterError(
                f'Visual output directory does not exist: "{search_dir}"'
            )

        safe_sensor = self._safe_sensor_name(sensor)
        # Match both regular frames and panorama frames:
        #   regular:  step{N}_{sensor}.png
        #   panorama: pano_{dir}_step{N}_{sensor}.png
        # Also accept legacy filenames with session_id prefix for compat.
        # Note: {N} is capture_seq, not simulator step_count.
        pattern = re.compile(
            rf"^(?:{re.escape(session_id)}_)?(?:pano_(?:front|right|back|left)_)?step(\d{{6}})_{re.escape(safe_sensor)}\.png$"
        )
        frame_records: list[tuple[int, str]] = []
        for file_name in os.listdir(search_dir):
            match = pattern.match(file_name)
            if match is None:
                continue
            seq = int(match.group(1))
            if step_start is not None and seq < step_start:
                continue
            if step_end is not None and seq > step_end:
                continue
            frame_records.append((seq, os.path.join(search_dir, file_name)))
        frame_records.sort(key=lambda item: item[0])
        return frame_records

    def _encode_video_trace(
        self, frame_paths: list[str], video_path: str, fps: float
    ) -> None:
        if not frame_paths:
            raise HabitatAdapterError("Cannot encode a video trace with no frames")

        output_dir = os.path.dirname(video_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = ""
        frame_duration_s = 1.0 / fps
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".ffconcat",
                prefix="video_trace_",
                dir=output_dir,
                delete=False,
            ) as manifest:
                manifest.write("ffconcat version 1.0\n")
                for frame_path in frame_paths[:-1]:
                    manifest.write(f"file {self._quote_ffconcat_path(frame_path)}\n")
                    manifest.write(f"duration {frame_duration_s:.6f}\n")
                last_frame_path = frame_paths[-1]
                manifest.write(f"file {self._quote_ffconcat_path(last_frame_path)}\n")
                manifest.write(f"duration {frame_duration_s:.6f}\n")
                manifest.write(f"file {self._quote_ffconcat_path(last_frame_path)}\n")
                manifest_path = manifest.name

            last_error_output = ""
            for codec in ("libx264", "mpeg4"):
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-loglevel",
                            "error",
                            "-f",
                            "concat",
                            "-safe",
                            "0",
                            "-i",
                            manifest_path,
                            "-an",
                            "-c:v",
                            codec,
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            video_path,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    last_error_output = ""
                    break
                except subprocess.CalledProcessError as exc:
                    last_error_output = (exc.stderr or exc.stdout or "").strip()
                    if os.path.exists(video_path):
                        os.unlink(video_path)
            else:
                raise HabitatAdapterError(
                    "Failed to encode video trace with ffmpeg"
                    + (f": {last_error_output}" if last_error_output else "")
                )
        except FileNotFoundError as exc:
            raise HabitatAdapterError(
                "ffmpeg is required to export video traces but was not found"
            ) from exc
        finally:
            if manifest_path and os.path.exists(manifest_path):
                os.unlink(manifest_path)

        if not os.path.exists(video_path) or os.path.getsize(video_path) <= 0:
            raise HabitatAdapterError(
                f'Video trace export did not produce a valid file: "{video_path}"'
            )

    @staticmethod
    def _quote_ffconcat_path(path: str) -> str:
        escaped = path.replace("\\", "\\\\").replace("'", "'\\''")
        return f"'{escaped}'"

    def _export_visuals(
        self,
        observation: Optional[Mapping[str, Any]],
        output_dir: str,
        sensors: Optional[list[str]],
        depth_max: float,
        session_id: str,
        step_count: int,
        capture_seq: Optional[int] = None,
        filename_prefix: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        if observation is None:
            raise HabitatAdapterError("No observation available for visualization")

        if sensors is None:
            sensor_names: list[str] = []
            for key, value in observation.items():
                try:
                    array_value = self._to_numpy_array(value)
                except HabitatAdapterError:
                    continue
                if array_value.ndim in (2, 3):
                    sensor_names.append(key)
        else:
            sensor_names = sensors

        visuals: Dict[str, Dict[str, Any]] = {}
        for sensor_name in sensor_names:
            if sensor_name not in observation:
                visuals[sensor_name] = {
                    "ok": False,
                    "error": f'Observation key "{sensor_name}" not found',
                }
                continue

            try:
                raw_value = observation[sensor_name]
                np_array = self._to_numpy_array(raw_value)

                if np_array.ndim == 3 and np_array.shape[2] in (3, 4):
                    image = self._to_uint8_color(np_array)
                    mode = "RGB" if image.shape[2] == 3 else "RGBA"
                elif np_array.ndim == 2:
                    image = self._to_uint8_depth(np_array, depth_max=depth_max)
                    mode = "L"
                else:
                    raise HabitatAdapterError(
                        f'Unsupported shape for sensor "{sensor_name}": '
                        f"{list(np_array.shape)}"
                    )
                item = self._write_png_image(
                    image=image,
                    output_dir=output_dir,
                    name=sensor_name,
                    session_id=session_id,
                    step_count=step_count,
                    capture_seq=capture_seq,
                    filename_prefix=filename_prefix,
                )
                item["mode"] = mode
                visuals[sensor_name] = item
            except Exception as exc:  # noqa: BLE001
                visuals[sensor_name] = {
                    "ok": False,
                    "error": str(exc),
                }

        return visuals

