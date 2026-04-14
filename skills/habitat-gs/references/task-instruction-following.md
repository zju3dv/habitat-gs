# Instruction Following Task

Follow natural language instructions that may involve navigation, observation, interaction, and reporting.

## Input

- `--goal "go to the kitchen, check if there's a table, then report"`: open-ended natural language instruction

Instructions can be:
- Sequential steps: "first go to X, then do Y, finally Z"
- High-level goals: "patrol the apartment and report what you see"
- Conditional: "if you find a door, go through it; otherwise turn around"

## Output

- `status: reached` with `finding` containing a structured report of what was done and observed
- The report should address each part of the instruction

## Starting

```bash
# Navmesh mode
hab nav-loop start --task-type instruction_following --mode navmesh \
  --goal "go to the kitchen, check if there's a table, then go to the bedroom"

# Mapless mode
hab nav-loop start --task-type instruction_following --mode mapless \
  --goal "explore this room and describe everything you see" --depth
```

## Strategy

### Step 1: Decompose the instruction
- Read the goal_description carefully
- Break it into concrete, verifiable substeps
- Each substep should be specific: navigate_to_room, inspect, report, etc.
- Submit the substep plan via nav-loop update

### Step 2: Execute each substep
- For navigation substeps: use the appropriate mode's tools (navigate for navmesh, polar+depth for mapless)
- For observation substeps: use `hab look` + VLM from multiple angles, turn to see different parts of the area
- For verification substeps: ask VLM if the instruction condition is met
- Record findings in spatial_memory_append as you go

### Step 3: Report
- Compile observations from all substeps
- Address each part of the original instruction
- Set status="reached" with a structured finding

### VLM prompts for instruction following

- Room identification: `"What room is this? (kitchen/bedroom/bathroom/hallway/office)"`
- Instruction checking: `"Current instruction: '{substep}'. Does this scene show completion? YES/NO/PARTIAL"`
- Direction guidance: `"I need to find {target_room}. What direction looks promising?"`
- Scene description: `"Describe this room in detail: furniture, objects, doors, windows"`

## Handling ambiguity

- If the instruction is vague, make a reasonable interpretation and proceed
- If a substep seems impossible (e.g., "go to the kitchen" but no kitchen exists), mark it as not achievable and explain why
- If instructions conflict, follow the most recent/specific one

## Common issues

- Can't find a referenced room → explore all accessible areas, report what was found instead
- Instruction requires interaction (e.g., "open the door") → report that physical interaction is not supported, describe what you see
- Too many substeps → prioritize the most important ones, report partial completion if time runs out

## Capability requests

- "I need room segmentation to identify room boundaries from visual observation"
- "I need a floor plan or room layout to plan efficient multi-room navigation"
- "I need text grounding to map instruction phrases to visual elements"
- "I need memory of previous instructions to handle follow-up tasks"
