**Rule 5.6 — Spatial memory accumulation.**
After each look/panorama, include spatial_memory_append in your update_nav_status call:
  {"heading_deg": N, "scene_description": "brief summary", "room_label": "room name or unknown", "objects_detected": ["obj1", "obj2"]}