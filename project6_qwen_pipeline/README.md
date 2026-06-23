# Project 6 Qwen Pipeline

This package turns PartNet-Mobility objects into compact part-layout examples
and uses Qwen to propose a new object layout from a natural-language request.

The intended user flow is simple:

```text
user request
-> Qwen chooses the closest PartNet-Mobility category
-> matching processed examples are loaded
-> Qwen proposes labeled part bounding boxes
-> the JSON output is validated
```

The Qwen output is a coarse structural layout. Later pipeline stages can turn
those boxes into a TRELLIS/RePaint structural condition.

## VM Setup For Teammates

Each teammate logs into the cluster with their own account and requests a GPU:

```bash
ssh <username>@ml3d.vc.in.tum.de
salloc --gpus=1
```

The shared project files are under:

```text
/cluster/52/jonasclotten/shared/project6
```

The shared Qwen weights are already downloaded here:

```text
/cluster/52/jonasclotten/shared/project6/models/Qwen3-8B
```

The repo has already been cloned here:

```text
/cluster/52/jonasclotten/shared/project6/repo
```

To get the latest code after someone pushed changes, run:

```bash
cd /cluster/52/jonasclotten/shared/project6/repo
git pull
```

Each teammate still needs their own Python/Conda environment. The shared folder
contains model weights and project files, not a shared Conda environment.

## Setup

From this folder, install the package into the active environment:

```bash
python -m pip install -e .
```

For the Qwen step on the VM, the environment also needs `torch`, `transformers`,
and `accelerate`. The code uses this default shared model path:

```text
/cluster/52/jonasclotten/shared/project6/models/Qwen3-8B
```

## Normal VM Usage

The normal entry point is:

```bash
python run_interactive.py
```

It asks for one user request in the terminal, for example:

```text
Create a compact oven with two front doors.
```

Then it automatically:

1. writes the category prompt to `prompts/`,
2. runs Qwen and writes the category answer to `outputs/`,
3. validates the category,
4. loads matching processed layout examples from `dataset/processed/`,
5. writes the layout prompt to `prompts/`,
6. runs Qwen again and writes the layout answer to `outputs/`,
7. validates the final layout JSON.

The user should not normally call the lower-level prompt commands by hand.

Current smoke-test status: we currently have one processed oven example for
end-to-end testing. After the full PartNet-Mobility dataset is available,
`dataset/processed/` should contain many examples across the available
categories, and the same interactive script should continue to be the main
entry point.

## Folder Convention

```text
dataset/processed/   compact layout examples extracted from PartNet-Mobility
prompts/             prompt text files sent to Qwen
outputs/             raw Qwen response files
```

The generated data, prompts, and outputs are local artifacts and are ignored by
Git except for their README files.

## Dataset Preparation

The full PartNet-Mobility dataset is expected to provide raw object folders
containing files such as:

```text
meta.json
semantics.txt
mobility.urdf
textured_objs/*.obj
```

The raw extractor reads those files, calculates one normalized bounding box per
semantic part, and writes compact layout JSON records.

For one raw object:

```bash
project6-layouts extract-raw \
  --input dataset/public_raw/7179 \
  --output dataset/processed/7179.json
```

For the temporary Exercise 4 tutor subset, this package also has a compatibility
path that strips away the provided bbox meshes and keeps only original part
geometry:

```bash
project6-layouts prepare \
  --source /path/to/partnet_primitives \
  --output dataset/tutor_raw_like

project6-layouts extract \
  --input dataset/tutor_raw_like \
  --output dataset/processed/layouts.jsonl
```

The tutor-subset path is only a development fallback. The real project should
use the raw PartNet-Mobility data once it is available.

## Debug Commands

The lower-level commands are useful when debugging a single step:

```bash
project6-layouts build-category-prompt \
  --request "Create a chair with three legs" \
  --output prompts/category_chair_three_legs.txt

project6-layouts run-qwen \
  --prompt prompts/category_chair_three_legs.txt \
  --output outputs/category_chair_three_legs.json

project6-layouts validate-category-response \
  --input outputs/category_chair_three_legs.json
```

For layout generation:

```bash
project6-layouts build-prompt \
  --request "A compact oven with two front doors" \
  --category Oven \
  --examples dataset/processed/7179.json \
  --output prompts/layout_compact_oven.txt

project6-layouts run-qwen \
  --prompt prompts/layout_compact_oven.txt \
  --output outputs/layout_compact_oven.json

project6-layouts validate-response \
  --input outputs/layout_compact_oven.json \
  --category Oven
```

## Tests

```bash
python -m unittest discover -s tests -v
```
