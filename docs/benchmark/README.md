# HabitatGS-Benchmark v1

[Download the public dataset](https://huggingface.co/datasets/yyh929/HabitatGS-Benchmark).

| Task | Test episodes | Scenes | Target visits |
| --- | ---: | ---: | ---: |
| ObjectNav | 150 | 66 | 150 |
| Ordered Multi-ObjectNav | 150 | 58 | 525 |

The union is 70 scenes. Multi-ObjectNav has 38/37/37/38 episodes with 2/3/4/5
ordered targets. Instructions describe specific objects rather than only object
categories. Both tasks use `habitat_gs/objectnav/v1` episodes.

## What is available

The HF repository contains `objectnav/v1/`, `multi-objectnav/v1/`, evaluation
scripts, JSON Schema files, statistics, manifests and checksums. Approved
viewpoints and reference paths are included for the evaluator, not the policy.

**Scene binaries are not included.** The shared manifest specifies the exact
70 GS/navmesh pairs. InteriorGS source files must be obtained under the
[upstream terms](https://kloudsim-usa-cos.kujiale.com/InteriorGS/InteriorGS_Terms_of_Use.pdf).
A verified source-to-runtime conversion and distribution route for matching
navmeshes are not yet provided. The annotation release is therefore not a
complete simulator-ready scene bundle. Do not substitute an arbitrary navmesh
and assume identical scores.

## Download and verify annotations

Install `huggingface_hub` and `jsonschema`, then pin a dataset revision:

```python
from huggingface_hub import HfApi, snapshot_download

repo = "yyh929/HabitatGS-Benchmark"
revision = HfApi().dataset_info(repo, token=False).sha
print("Record this dataset revision:", revision)
snapshot_download(repo, repo_type="dataset", revision=revision,
                  local_dir="habitatgs-benchmark", token=False)
```

```sh
python habitatgs-benchmark/evaluation/verify.py habitatgs-benchmark
```

Add `--scene-root /path/to/scenes` when the matching runtime assets are available.
The verifier checks file hashes and all 300 episode schemas. Downloads do not
require access to an internal checkout or the private development dataset.

## Evaluation

Read the [evaluation contract and submission format](evaluation.md). Success is
navmesh geodesic distance **strictly less than 0.1 m** to an approved viewpoint.
Single requires STOP at the goal; ordered Multi requires visiting all target
regions in order, followed by STOP. Bbox membership alone is not success.

Use `scripts_gs/benchmark/score.py` in this repository or the identical
`evaluation/score.py` in the data bundle. The offline scorer consumes trusted
runtime trajectories; it does not run a model or prove that a submitted movement
was executed. No baseline performance is claimed by this data release.

The HF root manifest binds the evaluation script and dataset files. Preserve
that version together with sensor/action/budget settings in every experiment.
