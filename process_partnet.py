"""
Merge per-part OBJ files for each PartNet-Mobility sample into a single OBJ.

Uses only stdlib + numpy — no open3d required.

Input layout:
    {input_dir}/{category}/{object_id}/objs/*.obj

Output layout:
    {output_dir}/{category}/{object_id}/mesh.obj

Usage:
    python process_partnet.py \\
        --input_dir partnet_mobility \\
        --output_dir partnet_mobility_processed \\
        [--workers 8]
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def _load_obj(path):
    """Return (vertices [N,3], faces [M,3]) from an OBJ file (0-indexed faces)."""
    verts = []
    faces = []
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif line.startswith("f "):
                    # Handle v, v/vt, v/vt/vn, v//vn
                    tokens = line.split()[1:]
                    indices = [int(t.split("/")[0]) - 1 for t in tokens]
                    # Fan triangulation for quads/ngons
                    for i in range(1, len(indices) - 1):
                        faces.append((indices[0], indices[i], indices[i + 1]))
    except Exception:
        pass
    return verts, faces


def _write_obj(path, all_verts, all_faces):
    """Write merged vertices and faces to an OBJ file."""
    with open(path, "w") as f:
        for v in all_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in all_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def merge_sample(args):
    src_dir, dst_dir = args
    obj_dir = src_dir / "objs"
    if not obj_dir.exists():
        return str(src_dir), "skip: no objs/ dir"

    obj_files = sorted(obj_dir.glob("*.obj"))
    if not obj_files:
        return str(src_dir), "skip: no .obj files"

    out_file = dst_dir / "mesh.obj"
    if out_file.exists():
        return str(src_dir), "already done"

    try:
        merged_verts = []
        merged_faces = []

        for f in obj_files:
            verts, faces = _load_obj(f)
            offset = len(merged_verts)
            merged_verts.extend(verts)
            merged_faces.extend(
                (a + offset, b + offset, c + offset) for a, b, c in faces
            )

        if not merged_verts:
            return str(src_dir), "skip: empty mesh"

        dst_dir.mkdir(parents=True, exist_ok=True)
        _write_obj(out_file, merged_verts, merged_faces)
        return str(src_dir), "ok"

    except Exception as e:
        return str(src_dir), f"error: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="partnet_mobility")
    p.add_argument("--output_dir", default="partnet_mobility_processed")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    tasks = []
    for cat_dir in sorted(input_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name == "hssd-data":
            continue
        for obj_dir in sorted(cat_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            dst = output_dir / cat_dir.name / obj_dir.name
            tasks.append((obj_dir, dst))

    print(f"Found {len(tasks)} samples across "
          f"{len(set(t[0].parent.name for t in tasks))} categories.")

    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(merge_sample, t): t for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(tasks), desc="Merging"):
            src, status = fut.result()
            if status.startswith("error"):
                errors.append((src, status))

    print(f"\nDone. {len(errors)} errors.")
    for src, msg in errors[:20]:
        print(f"  {src}: {msg}")


if __name__ == "__main__":
    main()
