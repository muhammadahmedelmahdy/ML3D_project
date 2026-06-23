# Project 6 Qwen Pipeline

This prototype extracts normalized, labeled part bounding boxes without using
the bounding boxes supplied in Exercise 4.

The temporary input format contains only information also expected from the
full PartNet-Mobility dataset:

- object ID and category
- part IDs and labels
- original part geometry

## Setup

```powershell
python -m pip install -e .
```

For the Qwen step on the VM, the environment also needs `torch`, `transformers`,
and `accelerate`. The default model path is:

```text
/cluster/52/jonasclotten/shared/project6/models/Qwen3-8B
```

## Folder convention

Keep prompt files and model answers separate:

```text
prompts/   text prompts that we send to Qwen
outputs/   raw answers that Qwen writes back
```

The output files are later checked by the validators.

## Prepare the tutor subset

Download or copy the Exercise 4 `partnet_primitives` directory somewhere
outside this project. Then strip it into the raw-like format:

```powershell
project6-layouts prepare `
  --source C:\path\to\partnet_primitives `
  --output dataset\tutor_raw_like
```

Only `parts_orig` meshes and sanitized metadata are copied. The command never
copies `parts_bbox`.

## Extract layouts

```powershell
project6-layouts extract `
  --input dataset\tutor_raw_like `
  --output dataset\processed\layouts.jsonl
```

Each output line contains one object with normalized Z-up part boxes in the
TRELLIS coordinate cube `[-0.5, 0.5]^3`.

## Raw PartNet adapter status

The first raw-data adapter stage parses `meta.json`, `semantics.txt`, and
`mobility.urdf`. It maps semantic URDF links to their visual OBJ paths and
records parent-link relationships. Visual meshes can now be loaded, scaled,
offset, and combined in each link's local frame. Joint transforms are
accumulated through the URDF hierarchy to place every part in one common
zero-pose object frame. The `extract-raw` command then calculates one box per
semantic part and normalizes the complete object into `[-0.5, 0.5]^3`:

```powershell
project6-layouts extract-raw `
  --input dataset\public_raw\7179 `
  --output dataset\processed\7179.json
```

## Build and run the category prompt

First, build a prompt from only the user's request:

```powershell
project6-layouts build-category-prompt `
  --request "Create a chair with three legs" `
  --output prompts\category_chair_three_legs.txt
```

Run that prompt through Qwen on the VM:

```bash
project6-layouts run-qwen \
  --prompt prompts/category_chair_three_legs.txt \
  --output outputs/category_chair_three_legs.json
```

Qwen must return a category such as `{"category":"Chair"}`. Validate it:

```powershell
project6-layouts validate-category-response `
  --input outputs\category_chair_three_legs.json
```

The returned category is then used to select layout examples for the second
prompt.

## Build and run the layout prompt

The prompt formatter accepts one to five already-selected layout records. It
does not search the dataset yet.

```powershell
project6-layouts build-prompt `
  --request "A compact oven with two front doors" `
  --category Oven `
  --examples dataset\processed\7179.json `
  --output prompts\layout_compact_oven.txt
```

Run Qwen on the VM:

```bash
project6-layouts run-qwen \
  --prompt prompts/layout_compact_oven.txt \
  --output outputs/layout_compact_oven.json \
  --max-new-tokens 2048
```

## Validate Qwen layout output

The response validator accepts only the requested JSON structure. It rejects
Markdown, missing or extra fields, duplicate IDs, invalid boxes, and
coordinates outside the normalized cube.

```powershell
project6-layouts validate-response `
  --input outputs\layout_compact_oven.json `
  --category Oven
```

## Tests

```powershell
python -m unittest discover -s tests -v
```
