**Rule 5.5 — Structured reasoning in action_history.**
Every action_history_append entry in update_nav_status MUST include three reasoning fields:
  - perception: what you actually see in the image/depth data (objects, rooms, doorways, obstacles, distances). Be specific and factual.
  - analysis: your interpretation — what does this mean for the task? Is the target visible? Which direction is promising?
  - decision: what you decided to do and WHY. State the chosen action and the reasoning.
Example:
  {"perception": "Office room with desk and monitor on left. Open doorway ahead-right leading to hallway. Depth shows 3.2m clearance ahead.",
   "analysis": "Target is a kitchen. This looks like an office. The doorway ahead-right likely leads to a hallway connecting to other rooms.",
   "decision": "Turn right 45° to face the doorway, then move forward to enter the hallway and search for the kitchen."}
Do NOT use one-line summaries. Each field should be 1-3 sentences with specific details.