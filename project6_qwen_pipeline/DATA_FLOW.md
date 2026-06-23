# Data Flow

This file explains what happens to the data before and during generation.

The pipeline has two parts:

1. **Offline preparation:** run once on the PartNet-Mobility dataset and create
   small example files.
2. **Online generation:** run whenever a user asks for a new object.

The important idea is:

```text
raw 3D dataset objects
-> simple part boxes
-> Qwen uses those boxes as examples
-> Qwen proposes new boxes for a new object
```

The user request does **not** directly load raw 3D meshes. The raw meshes are
processed beforehand into compact JSON files.

## Offline Preparation

### 1. Download PartNet-Mobility

PartNet-Mobility contains many 3D objects. Each object has its own folder. For
example, one folder may be one chair and another folder may be one oven.

Each object folder contains files such as:

- `meta.json`: says the object category, for example `Chair` or `Oven`.
- `semantics.txt`: gives readable part names, for example `leg`, `seat`, or
  `door`.
- `mobility.urdf`: describes how the parts belong together and where their mesh
  files are.
- `textured_objs/*.obj`: contains the actual 3D surface geometry.

These files come from the downloaded dataset. We do not create them ourselves.

### 2. Read one raw object

One raw object is one complete object, such as one specific oven. That oven may
contain several part instances:

- three trays
- two doors
- one oven body

The dataset usually gives these parts technical names such as `link_0`,
`link_1`, and `link_2`.

`semantics.txt` connects those technical names to readable labels:

```text
link_0 slider translation_tray
link_3 hinge door
link_5 static oven_body
```

`mobility.urdf` uses the same technical names and tells us which OBJ mesh files
belong to each part:

```text
link_3
-> textured_objs/original-1.obj
-> textured_objs/original-11.obj
-> textured_objs/original-29.obj
```

An OBJ file stores part of a 3D surface as points and triangles. One logical
part can use several OBJ files, so we first need to know which pieces belong
together.

`load_raw_partnet_object()` connects the files:

```text
semantics.txt says: link_3 is a door
URDF says: link_3 uses these OBJ files

result:
part instance link_3
label = door
geometry files = original-1.obj, original-11.obj, original-29.obj, ...
```

At this stage we only know the mapping. We have not loaded the 3D points yet.

```text
raw PartNet files
-> part names and lists of mesh files
```

### 3. Build each part's geometry

To calculate a box for one part, we need all 3D points of that part together.

`load_part_link_vertices()` does this for one part:

1. Load the XYZ points from every OBJ file assigned to that part.
2. Apply the position, rotation, and scale written in the URDF.
3. Combine all transformed points into one point cloud for that part.

Example:

```text
door OBJ piece 1: 800 points
door OBJ piece 2: 500 points
door OBJ piece 3: 300 points

-> one door with 1,600 points
```

The function does not generate new geometry. It only combines the geometry that
already exists in the dataset.

```text
several OBJ pieces for link_3
-> one combined door part
```

### 4. Put all parts into one object space

After step 3, each part has its own points. But those points may still be in the
part's local coordinate system.

Simple intuition: a door can describe itself relative to its own hinge. But for
the complete oven, we need to know where that door is relative to the oven body.

The URDF stores parent-child relationships such as:

```text
base
-> oven body
   -> door
   -> tray
```

It also stores the position and rotation between parent and child parts.

`link_transforms_to_root()` follows this chain. For a door, it combines:

```text
where the oven body is relative to the base
+ where the door is relative to the oven body
= where the door is in the complete object
```

`load_object_part_vertices()` applies these transforms to every part.

After this step, all parts use the same coordinate system. Now we can correctly
compare their positions, for example whether one tray is above another tray.

```text
parts in separate local coordinates
-> all parts in one shared object coordinate system
```

The current implementation uses the URDF **zero pose**. This means movable
parts such as doors and drawers stay in the default dataset position. We do not
open or move them ourselves.

### 5. Calculate normalized bounding boxes

Qwen does not need thousands of 3D points. It only needs a simple box for each
part.

For one part, `build_layout_record()` looks at all its points and finds:

- smallest X, Y, and Z values: `min`
- largest X, Y, and Z values: `max`

Those two corners define an axis-aligned bounding box. Axis-aligned means the
box is not rotated; it follows the X, Y, and Z axes.

Example:

```text
door points range from:
X: -0.8 to -0.6
Y: -0.5 to  0.5
Z: -0.7 to  0.4

door box:
min = [-0.8, -0.5, -0.7]
max = [-0.6,  0.5,  0.4]
```

Dataset objects can have very different sizes. Therefore, `build_layout_record()`
centers and uniformly scales the **complete object** so that it fits inside
`[-0.5, 0.5]^3`.

We scale the complete object once, not every part separately. This keeps the
relative layout intact. A door that was beside the body stays beside the body.

The output is Z-up, meaning Z is the height direction.

`extract_raw_partnet_object()` runs all raw-object steps for one object:

```text
raw object folder
-> positioned part geometry
-> labeled bounding boxes
-> compact layout JSON
```

Offline preparation repeats this for many dataset objects. The result is a
collection of compact JSON records in `dataset/processed/`. After that, Qwen no
longer needs the raw OBJ or URDF files for prompting.

## Online Generation

The normal online entry point is:

```bash
python run_interactive.py
```

The script asks for one text request in the terminal.

### 6. Choose the target category

Example request:

```text
Create a chair with three legs.
```

`build_category_prompt()` gives Qwen the request and the valid dataset category
names. It does not include geometry or bounding boxes yet.

Qwen returns a small JSON answer such as:

```json
{"category": "Chair"}
```

`parse_category_response()` checks that the answer is valid JSON and that
`Chair` is one of the allowed categories.

```text
user request + allowed categories
-> Qwen category prompt
-> Qwen category answer
-> validated category
```

### 7. Load matching examples

After the category is known, `load_layout_examples()` searches
`dataset/processed/` for objects with the same category.

For example, if Qwen selected `Chair`, it loads a few processed chair examples.
These examples contain only part labels and boxes, not raw meshes.

```text
validated category
-> matching processed examples
```

Current smoke-test status: we currently have one processed oven example. After
the full dataset is processed, this same step should load examples across many
categories.

### 8. Generate and validate a new layout

`build_layout_prompt()` gives Qwen:

- the original user request
- the validated category
- up to five matching example layouts

Qwen then proposes a new JSON layout with labeled part boxes.

`parse_layout_response()` checks that the answer has valid JSON, part IDs,
labels, box corners, category, and coordinate ranges.

```text
user request + category + examples
-> Qwen layout prompt
-> Qwen layout answer
-> validated new layout JSON
```

The validated layout is the output of the current Qwen-side pipeline. Later
TRELLIS/RePaint code can use these boxes as the rough structure that should be
kept or filled in during 3D generation.

## Complete Flow

```text
OFFLINE, ONCE:
downloaded PartNet objects
-> read part names and mesh files
-> place all parts in one object space
-> calculate normalized part boxes
-> save compact examples

ONLINE, FOR EACH USER REQUEST:
user text request
-> Qwen selects category
-> load matching examples
-> Qwen proposes new part boxes
-> validate layout JSON
-> pass boxes to later TRELLIS/RePaint stage
```
