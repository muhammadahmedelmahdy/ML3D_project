# Dataset Directory

This folder holds local dataset artifacts. Generated data is ignored by Git.

Expected subfolders:

```text
public_raw/        small raw PartNet-style samples for local smoke tests
tutor_raw_like/    temporary raw-like data derived from the Exercise 4 subset
processed/         compact layout JSON records used as Qwen examples
```

`processed/` is the important folder for the online pipeline. After the full
PartNet-Mobility dataset is available, preprocessing should fill it with many
layout examples across the available categories.

The temporary `tutor_raw_like/` path must contain only original part geometry
and sanitized labels/IDs. Provided Exercise 4 bbox meshes should not be placed
there, because the project pipeline should compute boxes itself.
