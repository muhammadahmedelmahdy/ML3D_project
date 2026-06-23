from __future__ import annotations

import numpy as np
import trimesh

from .models import RawJointSource, RawMeshSource, RawObjectSource, RawPartSource


def _load_mesh_vertices(source: RawMeshSource) -> np.ndarray:
    loaded = trimesh.load(source.mesh_path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise ValueError(f"Mesh is empty or unsupported: {source.mesh_path}")

    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"Mesh has invalid vertices: {source.mesh_path}")

    scaled = vertices * np.asarray(source.scale, dtype=np.float64)
    roll, pitch, yaw = source.origin_rpy
    rotation = trimesh.transformations.euler_matrix(roll, pitch, yaw, axes="sxyz")[:3, :3]
    rotated = scaled @ rotation.T
    return rotated + np.asarray(source.origin_xyz, dtype=np.float64)


def load_part_link_vertices(part: RawPartSource) -> np.ndarray:
    """Load and combine a part's visual meshes in its URDF link frame.

    Visual origin and scale are applied here. Parent-joint transforms are not;
    placing links in the complete object frame is the next preprocessing step.
    """
    if not part.meshes:
        raise ValueError(f"Part {part.link_name!r} has no meshes")
    return np.concatenate([_load_mesh_vertices(mesh) for mesh in part.meshes], axis=0)


def _pose_matrix(
    xyz: tuple[float, float, float], rpy: tuple[float, float, float]
) -> np.ndarray:
    roll, pitch, yaw = rpy
    matrix = trimesh.transformations.euler_matrix(roll, pitch, yaw, axes="sxyz")
    matrix[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return matrix


def link_transforms_to_root(raw_object: RawObjectSource) -> dict[str, np.ndarray]:
    """Compute every URDF link pose in the object's zero-pose root frame."""
    joints_by_child: dict[str, RawJointSource] = {}
    link_names: set[str] = {part.link_name for part in raw_object.parts}

    for joint in raw_object.joints:
        if joint.child_link in joints_by_child:
            raise ValueError(f"URDF link {joint.child_link!r} has more than one parent")
        joints_by_child[joint.child_link] = joint
        link_names.add(joint.parent_link)
        link_names.add(joint.child_link)

    transforms: dict[str, np.ndarray] = {}

    def resolve(link_name: str, active: set[str]) -> np.ndarray:
        if link_name in transforms:
            return transforms[link_name]
        if link_name in active:
            raise ValueError(f"Cycle detected in URDF hierarchy at {link_name!r}")

        joint = joints_by_child.get(link_name)
        if joint is None:
            transform = np.eye(4, dtype=np.float64)
        else:
            parent_transform = resolve(joint.parent_link, active | {link_name})
            local_transform = _pose_matrix(joint.origin_xyz, joint.origin_rpy)
            transform = parent_transform @ local_transform

        transforms[link_name] = transform
        return transform

    for link_name in link_names:
        resolve(link_name, set())
    return transforms


def load_object_part_vertices(raw_object: RawObjectSource) -> dict[str, np.ndarray]:
    """Load semantic parts and place them in one common zero-pose frame."""
    transforms = link_transforms_to_root(raw_object)
    object_vertices: dict[str, np.ndarray] = {}

    for part in raw_object.parts:
        link_vertices = load_part_link_vertices(part)
        transform = transforms[part.link_name]
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        object_vertices[part.link_name] = link_vertices @ rotation.T + translation

    return object_vertices
