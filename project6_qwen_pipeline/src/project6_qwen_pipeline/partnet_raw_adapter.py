from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .models import RawJointSource, RawMeshSource, RawObjectSource, RawPartSource


@dataclass(frozen=True)
class SemanticEntry:
    link_name: str
    joint_type: str
    label: str


def parse_semantics(path: Path) -> tuple[SemanticEntry, ...]:
    """Parse `link_name joint_type label` rows from semantics.txt."""
    entries: list[SemanticEntry] = []
    seen_links: set[str] = set()

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue

        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"Invalid semantics row at {path}:{line_number}")

        link_name, joint_type, label = fields
        if link_name in seen_links:
            raise ValueError(f"Duplicate semantic link {link_name!r} in {path}")

        seen_links.add(link_name)
        entries.append(SemanticEntry(link_name, joint_type, label))

    if not entries:
        raise ValueError(f"No semantic entries found in {path}")
    return tuple(entries)


def _safe_mesh_path(object_directory: Path, filename: str) -> Path:
    relative_path = Path(filename)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe URDF mesh path: {filename}")
    return object_directory / relative_path


def _parse_vector(
    value: str | None,
    default: tuple[float, float, float],
    field_name: str,
) -> tuple[float, float, float]:
    if value is None or not value.strip():
        return default

    fields = value.split()
    if len(fields) != 3:
        raise ValueError(f"URDF {field_name} must contain three numbers: {value!r}")
    try:
        return tuple(float(field) for field in fields)  # type: ignore[return-value]
    except ValueError as error:
        raise ValueError(f"URDF {field_name} contains a non-number: {value!r}") from error


def _visual_meshes_by_link(
    root: ET.Element, object_directory: Path
) -> dict[str, tuple[RawMeshSource, ...]]:
    meshes_by_link: dict[str, tuple[RawMeshSource, ...]] = {}

    for link in root.findall("link"):
        link_name = str(link.get("name", "")).strip()
        if not link_name:
            raise ValueError("URDF contains a link without a name")

        sources: list[RawMeshSource] = []
        seen_paths: set[Path] = set()
        for visual in link.findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            filename = str(mesh.get("filename", "")).strip()
            if not filename:
                raise ValueError(f"Visual mesh without filename in link {link_name!r}")
            mesh_path = _safe_mesh_path(object_directory, filename)
            if mesh_path not in seen_paths:
                seen_paths.add(mesh_path)
                origin = visual.find("origin")
                sources.append(
                    RawMeshSource(
                        mesh_path=mesh_path,
                        origin_xyz=_parse_vector(
                            origin.get("xyz") if origin is not None else None,
                            (0.0, 0.0, 0.0),
                            "origin xyz",
                        ),
                        origin_rpy=_parse_vector(
                            origin.get("rpy") if origin is not None else None,
                            (0.0, 0.0, 0.0),
                            "origin rpy",
                        ),
                        scale=_parse_vector(
                            mesh.get("scale"),
                            (1.0, 1.0, 1.0),
                            "mesh scale",
                        ),
                    )
                )

        meshes_by_link[link_name] = tuple(sources)

    return meshes_by_link


def _parse_joints(root: ET.Element) -> tuple[RawJointSource, ...]:
    joints: list[RawJointSource] = []
    seen_children: set[str] = set()

    for joint in root.findall("joint"):
        child_element = joint.find("child")
        parent_element = joint.find("parent")
        if child_element is None or parent_element is None:
            raise ValueError("URDF joint is missing a parent or child")

        child = str(child_element.get("link", "")).strip()
        parent = str(parent_element.get("link", "")).strip()
        if not child or not parent:
            raise ValueError("URDF joint has an empty parent or child link")
        if child in seen_children:
            raise ValueError(f"URDF link {child!r} has more than one parent")
        seen_children.add(child)

        origin = joint.find("origin")
        joints.append(
            RawJointSource(
                name=str(joint.get("name", "")).strip(),
                joint_type=str(joint.get("type", "")).strip(),
                parent_link=parent,
                child_link=child,
                origin_xyz=_parse_vector(
                    origin.get("xyz") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                    "joint origin xyz",
                ),
                origin_rpy=_parse_vector(
                    origin.get("rpy") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                    "joint origin rpy",
                ),
            )
        )

    return tuple(joints)


def load_raw_partnet_object(
    object_directory: Path,
    *,
    require_mesh_files: bool = True,
) -> RawObjectSource:
    """Map raw PartNet metadata, semantics and URDF links to labeled parts.

    This stage only identifies part-to-mesh relationships. It deliberately
    does not load geometry or apply URDF transforms yet.
    """
    object_directory = Path(object_directory)
    meta_path = object_directory / "meta.json"
    semantics_path = object_directory / "semantics.txt"
    urdf_path = object_directory / "mobility.urdf"

    for required_path in (meta_path, semantics_path, urdf_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing raw PartNet file: {required_path}")

    with meta_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    object_id = str(metadata.get("anno_id", object_directory.name)).strip()
    category = str(metadata.get("model_cat", "")).strip()
    if not object_id or not category:
        raise ValueError(f"Missing object ID or category in {meta_path}")

    root = ET.parse(urdf_path).getroot()
    meshes_by_link = _visual_meshes_by_link(root, object_directory)
    joints = _parse_joints(root)
    parents_by_child = {joint.child_link: joint.parent_link for joint in joints}

    parts: list[RawPartSource] = []
    for semantic in parse_semantics(semantics_path):
        if semantic.link_name not in meshes_by_link:
            raise ValueError(
                f"Semantic link {semantic.link_name!r} is absent from {urdf_path}"
            )

        mesh_sources = meshes_by_link[semantic.link_name]
        if not mesh_sources:
            raise ValueError(f"Semantic link {semantic.link_name!r} has no visual meshes")
        if require_mesh_files:
            missing = [
                source.mesh_path for source in mesh_sources if not source.mesh_path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Semantic link {semantic.link_name!r} references missing mesh {missing[0]}"
                )

        parts.append(
            RawPartSource(
                link_name=semantic.link_name,
                label=semantic.label,
                semantic_joint_type=semantic.joint_type,
                parent_link=parents_by_child.get(semantic.link_name),
                meshes=mesh_sources,
            )
        )

    return RawObjectSource(object_id, category, tuple(parts), joints)
