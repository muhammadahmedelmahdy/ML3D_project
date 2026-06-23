# Data Flow

There are two separate flows:

1. **Offline preparation:** process PartNet-Mobility once and create compact
   layout examples.
2. **Online generation:** receive a user request, select relevant prepared
   examples, and ask Qwen for a new layout.

The user request does **not** load raw 3D assets directly.

## Offline Preparation

### 1. Download PartNet-Mobility

The full PartNet-Mobility dataset is downloaded before the pipeline runs. It
contains one directory per object. For example, one directory may represent a
chair and another may represent an oven.

Each raw object directory provides files such as:

- `meta.json`: says whether the object is a chair, oven, etc.
- `semantics.txt`: gives names such as `leg`, `seat`, or `door`.
- `mobility.urdf`: connects parts, transforms, and OBJ paths.
- `textured_objs/*.obj`: contains the actual 3D geometry.

These files come from the downloaded dataset. We do not create them.

### 2. Read one raw PartNet object

One PartNet object is one complete object, such as one specific oven. That oven
contains several **part instances**, for example:

- three trays
- two doors
- one oven body

The dataset does not initially call them `door_1` or `tray_2`. Instead, the
URDF gives every part instance a technical link name such as `link_0`.

`semantics.txt` connects the technical name to a human-readable label:

```text
link_0 slider translation_tray
link_3 hinge door
link_5 static oven_body
```

`mobility.urdf` contains the same technical link names. Inside each link, it
lists the OBJ geometry files used to draw that part:

```text
link_3
-> textured_objs/original-1.obj
-> textured_objs/original-11.obj
-> textured_objs/original-29.obj
```

An OBJ file stores a piece of a 3D surface as points and triangles. One logical
part can use several OBJ files because its surface may have been saved as
separate pieces. These pieces still belong to the same part instance.

`load_raw_partnet_object()` matches the repeated link name across the files:

```text
semantics.txt says: link_3 is a door
URDF says: link_3 uses these OBJ files

result:
part instance link_3
label = door
geometry files = original-1.obj, original-11.obj, original-29.obj, ...
```

This function only creates the mapping. It does not load the 3D points yet.

```text
raw PartNet files
-> part instances with names and lists of geometry files
```

### 3. Build each part's geometry

Each OBJ file contains many XYZ points and triangles. To calculate one box for
a complete door, we first need all surface pieces of that door together.

`load_part_link_vertices()` does this for one part instance:

1. Load the XYZ points from every OBJ file assigned to the part.
2. Apply the position, rotation, and size stored for each piece in the URDF.
3. Combine all transformed points into one collection for that part.

Example:

```text
door OBJ piece 1: 800 points
door OBJ piece 2: 500 points
door OBJ piece 3: 300 points

-> one door containing 1,600 points
```

The function does not invent or generate new geometry. It only combines the
pieces that the dataset says belong together.

```text
several OBJ surface pieces for link_3
-> one combined door part
```

### 4. Place all parts in the object

After step 3, we know the shape of each part, but its points can still be in the
part's own **local coordinate system**.

For example, a door may initially think that its own hinge is at position
`[0, 0, 0]`. That does not mean the hinge belongs at the center of the complete
oven. We must still place the door relative to the oven body.

The URDF stores parent-child relationships such as:

```text
base
-> oven body
   -> door
   -> tray
```

It also stores the position and rotation between each parent and child.

`link_transforms_to_root()` follows this chain. For a door, it combines:

```text
where the oven body is relative to the base
+ where the door is relative to the oven body
= where the door is in the complete object
```

`load_object_part_vertices()` applies these transforms to every part.

Afterward, all parts use the same coordinate system. We can now correctly say
that one tray is above another tray or that a door is in front of the body.

```text
separate parts in local coordinates
-> all parts in one object coordinate system
```

The current implementation uses the URDF **zero pose**. This means movable
parts such as doors and drawers stay in the default position recorded by the
dataset. We are not opening or moving them ourselves.

### 5. Calculate normalized bounding boxes

Now every part is represented by many XYZ surface points. Qwen does not need
all these points. It only needs a simple box describing where the part is.

For one part, `build_layout_record()` looks through all its points and finds:

- smallest X, Y, and Z values: `min`
- largest X, Y, and Z values: `max`

These two corners define the smallest axis-aligned box containing the part.

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

Different dataset objects can have very different real sizes and coordinate
values. Therefore, `build_layout_record()` centers and uniformly scales the
**complete object** so that it fits inside `[-0.5, 0.5]^3`.

We normalize the complete object once, not every part separately. This keeps
the relative layout intact. A door that was beside the body remains beside the
body after normalization.

The output is Z-up, meaning the Z direction represents height.

`extract_raw_partnet_object()` runs the reading, placement, and box-calculation
steps for one raw object.

```text
positioned part geometry
-> labeled numerical bounding boxes
-> canonical layout JSON
```

Example output:

```json
{
  "category": "Oven",
  "parts": [
    {
      "id": "link_3",
      "label": "door",
      "bbox": {
        "min": [-0.30, -0.35, -0.48],
        "max": [-0.24, 0.35, 0.05]
      }
    }
  ]
}
```

The offline preparation repeats these steps for many dataset objects. The
result is a collection of compact layout JSON records. Qwen does not need the
raw OBJ or URDF files after this preparation.

## Online Generation

### 6. Receive a user request

Example:

```text
Create a chair with three legs.
```

`build_category_prompt()` combines the request with the 46 valid dataset
category names. No geometry or bounding boxes are included in this first Qwen
prompt.

Qwen returns a small response such as:

```json
{"category": "Chair"}
```

`parse_category_response()` checks that this is valid JSON and that `Chair` is
one of the allowed dataset categories. A retrieval step will then select a few
prepared chair layout records from the collection.

```text
user request + list of allowed categories
-> first Qwen prompt
-> validated target category
-> relevant prepared layout examples
```

The retrieval function is not implemented yet. For now, examples are supplied
manually.

### 7. Build the Qwen prompt

`build_layout_prompt()` receives one or more already-selected layout records.
It converts up to five examples into compact JSON and adds the user's request.

```text
example layouts + user request
-> Qwen prompt
```

Currently, the prompt uses the one public oven example. Dataset retrieval is
not implemented yet.

### 8. Generate and validate a new layout

Qwen receives the prompt and returns a new description with labeled bounding
boxes. The actual Qwen model call is not implemented in this project yet.

`parse_layout_response()` validates the returned JSON, part IDs, labels, box
corners, category, and coordinate range.

```text
Qwen prompt
-> Qwen response text
-> validated new layout JSON
```

## Complete Flow

```text
OFFLINE, ONCE:
downloaded PartNet objects
-> named and positioned parts
-> normalized bounding boxes
-> collection of prepared layout examples

ONLINE, FOR EACH USER REQUEST:
user text request
-> select relevant prepared examples
-> Qwen prompt
-> proposed bounding boxes
-> validated layout JSON
```
