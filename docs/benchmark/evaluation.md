# ObjectNav and ordered Multi-ObjectNav evaluation

This evaluator implements the public v1 evaluation contract (`habitat_gs/objectnav/v1`).
Install Habitat-GS first. No model or internal agent framework is required for scoring.

Success uses **navmesh geodesic distance strictly less than 0.1 m** to any
approved viewpoint of the active target. Euclidean distance, bbox membership,
and distance to a single reference endpoint are not substitutes.

- Single ObjectNav: execute STOP while inside the goal region.
- Ordered Multi-ObjectNav: visit all target regions in order, then execute STOP.
  Earlier goals need not remain close to the terminal position. Overlapping
  consecutive regions can be visited at the same recorded pose.
- The first STOP, budget exhaustion, or runtime error terminates the record;
  later events are ignored. Reaching a goal without STOP does not succeed.

## Submission format

The trusted runtime records initial agent position and actual post-action agent
position after **every primitive action**. These are agent positions, not camera
positions or intended waypoints. Do not let the policy fabricate the trajectory.
The first `start` must match the published initial position (within 0.1 mm to
allow serialization precision). STOP does not translate the agent.

```json
{
  "episodes": [
    {
      "episode_id": "<published episode ID>",
      "events": [
        {"action": "start", "position": [0.0, 0.0, 0.0]},
        {"action": "move_forward", "position": [0.0, 0.0, -0.25]},
        {"action": "stop", "position": [0.0, 0.0, -0.25]}
      ]
    }
  ]
}
```

The coordinates above demonstrate formatting only; they are not a benchmark
episode. Runtime failures use `{"action": "runtime_error"}`; reaching the
declared experiment limit uses `{"action": "budget_exhausted"}`.

Duplicate/unknown IDs and malformed trajectories are rejected. Unsubmitted
episodes count as failures. Runtime errors have null success/SPL and are
excluded from evaluated means; always report `total`, `evaluated`, and
`runtime_errors` alongside SR/SPL, never the successful subset alone.

## Run

```sh
python scripts_gs/benchmark/score.py \
  --episodes /path/to/objectnav/v1/episodes.json \
  --submissions /path/to/submissions.json \
  --scene-root /path/to/scenes \
  --reference global_shortest \
  --output /path/to/new-scores.json
```

Use the Multi-ObjectNav episode file for ordered navigation. The HF bundle
exposes the same script as `evaluation/score.py`. The scorer needs paired
navmeshes; rendering additionally requires the GS files. Never substitute a
newly baked navmesh without verifying equivalence to the published scene manifest.

SPL is `success * L / max(L, P)`, with value 1 for successful zero-length
reference and execution. `L` is the published `global_shortest` reference
length by default, optimal over retained viewpoint sets. `local_shortest`
is available explicitly for the greedy reference; report the selected variant.
`P` is the sum of distances between recorded primitive-action agent positions.
Sparse waypoint logs are insufficient. This offline scorer cannot authenticate
movement or recover missing intermediate samples: the runtime must execute and
record actual collision-aware motion. Scripted GT checks are not baseline results.

The annotations do not specify one universal sensor/action/budget configuration.
Publish sensor settings, primitive step/turn sizes, all budget limits, runtime
revision, and dataset revision with results. Do not claim scores from different
configurations are controlled comparisons.

Viewpoints, bboxes, reference paths, navmesh queries, and audit evidence stay on
the evaluator side. The policy receives instructions and actual sensor observations.
