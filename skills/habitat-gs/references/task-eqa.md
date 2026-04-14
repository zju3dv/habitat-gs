# EQA Task (Embodied Question Answering)

Answer a question about the environment by observing and, if needed, actively exploring to gather evidence.

## Input

- `--goal "Is there a microwave in this scene? Answer yes or no."`: the question to answer

The question format determines the expected answer format:
- Yes/no questions → answer `yes`, `no`, or `unknown`
- Multiple choice → select the best option
- Open-ended → concise natural language answer

## Output

- `status: reached` with `finding` containing the answer, evidence, and confidence
- Expected finding format:
  ```json
  {"answer": "yes", "evidence": "Microwave visible on the kitchen counter", "confidence": "high"}
  ```

## Starting

```bash
# Navmesh mode (can explore efficiently with map tools)
hab nav-loop start --task-type eqa --mode navmesh --goal "Is there a microwave in this scene? Answer yes or no."

# Mapless mode
hab nav-loop start --task-type eqa --mode mapless --goal "How many chairs are in this room?" --depth
```

## Strategy

### Key principle: observe first, explore only if needed

Unlike navigation tasks, the goal is to ANSWER A QUESTION, not reach a location. Movement is only a means to gather evidence.

### Step 1: Observe and judge
- `hab look` to see the current view
- Ask VLM: "Question: {question}. Can you answer from what you see? If YES, provide your answer. If NO, what evidence do you need?"
- If answerable with confidence → skip to answer immediately

### Step 2: Explore for evidence (if needed)
- Think about where the relevant evidence might be (e.g., microwave → kitchen, chairs → dining area)
- Use `hab panorama` to scan all directions
- Navigate toward promising areas
- After each new viewpoint, re-evaluate: "Can I answer now?"
- Use spatial_memory to track what you've checked

### Step 3: Answer
- Compile evidence from all observations
- Provide answer in the expected format
- Include confidence level (high/medium/low)
- If thoroughly explored and still uncertain, answer "unknown" with explanation

## Answer format guidelines

| Question type | Example | Expected answer |
|---|---|---|
| Yes/no | "Is there a TV?" | `yes` / `no` / `unknown` + evidence |
| Counting | "How many doors?" | Number + evidence (e.g., "2 doors: one in hallway, one in bedroom") |
| Location | "Where is the desk?" | Description (e.g., "In the office room, against the left wall") |
| Multiple choice | "What color is the sofa? (a) red (b) blue (c) brown" | Selected option + evidence |
| Open-ended | "Describe the kitchen" | Concise description based on observations |

## Common issues

- Question about an object not in current view → explore other rooms
- Ambiguous question → make reasonable interpretation, state assumption
- Multiple possible answers → provide the most confident one with evidence

## Capability requests

- "I need object counting capability to accurately count items in the scene"
- "I need semantic segmentation to identify room boundaries"
- "I need scene label data to verify object presence without visual search"
