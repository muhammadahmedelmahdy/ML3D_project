from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PartSource:
    part_id: str
    label: str
    mesh_path: Path


@dataclass(frozen=True)
class ObjectSource:
    object_id: str
    category: str
    parts: tuple[PartSource, ...]


@dataclass(frozen=True)
class RawMeshSource:
    mesh_path: Path
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class RawPartSource:
    link_name: str
    label: str
    semantic_joint_type: str
    parent_link: str | None
    meshes: tuple[RawMeshSource, ...]

    @property
    def mesh_paths(self) -> tuple[Path, ...]:
        return tuple(mesh.mesh_path for mesh in self.meshes)


@dataclass(frozen=True)
class RawJointSource:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]


@dataclass(frozen=True)
class RawObjectSource:
    object_id: str
    category: str
    parts: tuple[RawPartSource, ...]
    joints: tuple[RawJointSource, ...] = ()
