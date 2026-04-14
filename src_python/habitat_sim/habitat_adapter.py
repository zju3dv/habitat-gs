#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from .habitat_adapter_internal.mixins_api import HabitatAdapterApiMixin
from .habitat_adapter_internal.mixins_core import HabitatAdapterCoreMixin
from .habitat_adapter_internal.mixins_nav_loop import HabitatAdapterNavLoopMixin
from .habitat_adapter_internal.mixins_navigation import (
    HabitatAdapterNavigationMixin,
)
from .habitat_adapter_internal.mixins_patch import HabitatAdapterPatchMixin
from .habitat_adapter_internal.mixins_session_scene import (
    HabitatAdapterSessionSceneMixin,
)
from .habitat_adapter_internal.mixins_visual_media import (
    HabitatAdapterVisualMediaMixin,
)
from .habitat_adapter_internal.types import (
    SUPPORTED_ACTIONS as _SUPPORTED_ACTIONS,
    _API_VERSION,
    _DEFAULT_DEPTH_VIS_MAX,
    _DEFAULT_MAX_OBSERVATION_ELEMENTS,
    _DEFAULT_TOPDOWN_METERS_PER_PIXEL,
    _DEFAULT_VIDEO_FPS,
    _DEFAULT_VISUAL_OUTPUT_DIR,
    _MAX_CLOSED_NAV_LOOPS,
    _MAX_NAVIGATE_STEP_BURST,
    _MAX_NAV_ACTION_HISTORY,
    _NAV_LOOP_SCRIPT,
    _NAV_STATUS_ALLOWED_VALUES,
    _NAV_STATUS_IMMUTABLE_FIELDS,
    _NAV_STATUS_PATCH_FIELDS,
    _NavLoopRecord,
    _Session,
    HabitatAdapterError,
)


class HabitatAdapter(
    HabitatAdapterCoreMixin,
    HabitatAdapterApiMixin,
    HabitatAdapterSessionSceneMixin,
    HabitatAdapterNavigationMixin,
    HabitatAdapterVisualMediaMixin,
    HabitatAdapterPatchMixin,
    HabitatAdapterNavLoopMixin,
):
    """Composed habitat adapter facade."""

    SUPPORTED_ACTIONS = _SUPPORTED_ACTIONS


__all__ = ["HabitatAdapter", "HabitatAdapterError"]
