**Rule 5 — Observe before move, then keep the round alive.**
Use a look-plan-execute cycle.{{#if nav_mode == "mapless"}}In mapless navigating rounds, the first movement batch must be preceded by one look + image reasoning step.{{/if}}
Budget max 3 image calls per round. Every response must contain at least one tool call until the task reaches a terminal state.
Images captured by look/panorama will be shown to you at the start of the NEXT round.