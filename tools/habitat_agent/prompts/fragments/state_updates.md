**Rule 6 — State updates via update_nav_status tool.**
Use the update_nav_status tool with a patch containing ONLY mutable fields:
  substeps, current_substep_index, status, nav_phase, total_steps, collisions,
  last_action, action_history_append, spatial_memory_append, finding, error, capability_request
{{#if nav_mode == "mapless"}}
For mapless updates: include non-empty last_action, at least one action_history_append with perception/analysis/decision, and one spatial_memory_append.
{{/if}}