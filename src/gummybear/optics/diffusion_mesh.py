"""Coarse tetrahedral diffusion mesh generation from an STL triangle mesh file via Netgen."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

import tempfile


@dataclass(frozen=True)
class DiffusionMeshMetadata:
    """Serializable provenance and cache keys for a generated diffusion mesh.

    The diffusion mesh is derived geometry (non-authoritative relative to the
    surface STL triangle mesh file). ``cache_keys`` captures parameters used to locate on-disk
    npz/json artifacts.

    Attributes
    ----------
    stl_hash:
        Content hash of the source STL triangle mesh file.
    geometry_id:
        Path or in-memory identifier for the source geometry.
    units:
        Length unit label (default ``"mm"``).
    meshing_method:
        Always ``"netgen"`` for the implemented path.
    num_elements, num_nodes:
        Tet and node counts after meshing.
    target_elements, maxh, grading:
        Netgen sizing parameters used or estimated.
    is_authoritative_geometry:
        Always ``False`` — surface STL triangle mesh file remains canonical.
    resolution_basis:
        Rationale for mesh sizing (``"diffusion_length_scale"``).
    cache_keys:
        Full parameter dict for cache lookup.
    """

    stl_hash: str
    geometry_id: str
    units: str = "mm"
    meshing_method: str = "netgen"
    netgen_version: str | None = None
    num_elements: int = 0
    num_nodes: int = 0
    target_elements: int | None = None
    maxh: float | None = None
    grading: float | None = None
    is_authoritative_geometry: bool = False
    resolution_basis: str = "diffusion_length_scale"
    cache_keys: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata dictionary."""
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        """Write metadata to ``path`` as an indented JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


@dataclass
class DiffusionMesh:
    """Coarse tetrahedral mesh carrying volumetric diffusion geometry.

    Derived from the surface STL triangle mesh file for volumetric diffusion; not the authoritative
    rendering mesh. Retains an optional live ``netgen_mesh`` handle required
    by NGSolve finite-element method (FEM) assembly — cache-only reloads set this to ``None``.

    Attributes
    ----------
    nodes:
        Node coordinates, shape ``[n_nodes, 3]`` (mesh units).
    tets:
        Tetrahedron vertex indices (0-based), shape ``[n_tets, 4]``.
    centroids:
        Tet centroids, shape ``[n_tets, 3]``.
    volumes:
        Tet volumes, shape ``[n_tets]``.
    metadata:
        :class:`DiffusionMeshMetadata` provenance.
    netgen_mesh:
        Live Netgen mesh object when freshly generated; ``None`` when loaded
        from cache without a finite-element method (FEM) handle.

    See also:
        :func:`~gummybear.optics.diffusion_solve.solve_diffusion` — steady-state fluence solve on this mesh.
        :func:`~gummybear.optics.source_deposition.deposit_ray_source` — volumetric source deposition into tets.
    """

    nodes: np.ndarray  # [n_nodes, 3]
    tets: np.ndarray  # [n_tets, 4], 0-based
    centroids: np.ndarray  # [n_tets, 3]
    volumes: np.ndarray  # [n_tets]
    metadata: DiffusionMeshMetadata
    netgen_mesh: Any | None = None  # optional Netgen mesh object for NGSolve

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.tets = np.asarray(self.tets, dtype=int)
        self.centroids = np.asarray(self.centroids, dtype=float)
        self.volumes = np.asarray(self.volumes, dtype=float)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError(f"nodes must be [N, 3], got {self.nodes.shape}")
        if self.tets.ndim != 2 or self.tets.shape[1] != 4:
            raise ValueError(f"tets must be [M, 4], got {self.tets.shape}")
        if len(self.centroids) != len(self.tets) or len(self.volumes) != len(self.tets):
            raise ValueError("centroids/volumes length must match n_tets")

    @property
    def n_nodes(self) -> int:
        """Number of mesh nodes (``len(nodes)``)."""
        return len(self.nodes)

    @property
    def n_tets(self) -> int:
        """Number of tetrahedral elements (``len(tets)``)."""
        return len(self.tets)

    @property
    def bounds(self) -> np.ndarray:
        """Axis-aligned bounding box, shape ``[2, 3]`` (min/max)."""
        return np.vstack([self.nodes.min(axis=0), self.nodes.max(axis=0)])

    def content_hash(self) -> str:
        """SHA-256 hash of ``nodes`` and ``tets`` arrays for cache keys."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.nodes).tobytes())
        h.update(np.ascontiguousarray(self.tets).tobytes())
        return h.hexdigest()

    def save_npz(self, path: str | Path) -> None:
        """Persist nodes/tets/centroids/volumes and write sidecar metadata JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            nodes=self.nodes,
            tets=self.tets,
            centroids=self.centroids,
            volumes=self.volumes,
        )
        meta_path = path.with_suffix(".json")
        if meta_path.name.endswith(".npz.json"):
            meta_path = path.with_name(path.stem + "_meta.json")
        self.metadata.write_json(path.with_name(path.stem + "_meta.json"))


def _require_netgen():
    try:
        from netgen.stl import STLGeometry  # noqa: F401
        from netgen.meshing import MeshingParameters  # noqa: F401
        import netgen
    except ImportError as exc:
        raise ImportError(
            "M4A Netgen path requires the optional 'fem' extra "
            "(pip install '.[fem]'). SciPy/trimesh fallback is not implemented."
        ) from exc
    return netgen


def _stl_hash(mesh_or_path: trimesh.Trimesh | str | Path) -> tuple[str, str]:
    if isinstance(mesh_or_path, (str, Path)):
        path = Path(mesh_or_path)
        data = path.read_bytes()
        return hashlib.sha256(data).hexdigest(), str(path)
    verts = np.ascontiguousarray(mesh_or_path.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh_or_path.faces, dtype=np.int64)
    h = hashlib.sha256()
    h.update(verts.tobytes())
    h.update(faces.tobytes())
    return h.hexdigest(), "trimesh_in_memory"


def _estimate_maxh(bounds: np.ndarray, target_elements: int) -> float:
    extents = np.asarray(bounds[1] - bounds[0], dtype=float)
    vol = float(np.prod(np.maximum(extents, 1e-12)))
    return float((vol / max(int(target_elements), 1)) ** (1.0 / 3.0))


def _tet_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    v = nodes[tets]
    return (
        np.abs(
            np.einsum(
                "ij,ij->i",
                v[:, 0] - v[:, 3],
                np.cross(v[:, 1] - v[:, 3], v[:, 2] - v[:, 3]),
            )
        )
        / 6.0
    )


def _extract_nodes_tets(nmesh) -> tuple[np.ndarray, np.ndarray]:
    points = nmesh.Points()
    nodes: list[list[float]] = []
    i = 1
    while True:
        try:
            p = points[i]
        except IndexError:
            break
        nodes.append([float(p[0]), float(p[1]), float(p[2])])
        i += 1
    nodes_arr = np.asarray(nodes, dtype=float)

    tets = []
    for el in nmesh.Elements3D():
        ids = [int(v.nr) - 1 for v in el.vertices]
        tets.append(ids)
    tets_arr = np.asarray(tets, dtype=int)
    if tets_arr.size == 0:
        raise RuntimeError("Netgen produced zero volume tetrahedra")
    if tets_arr.min() < 0 or tets_arr.max() >= len(nodes_arr):
        raise RuntimeError(
            f"Tet vertex indices out of range: "
            f"[{tets_arr.min()}, {tets_arr.max()}] vs n_nodes={len(nodes_arr)}"
        )
    return nodes_arr, tets_arr


def _mesh_stl_to_tet_netgen(
    stl_path: Path,
    *,
    maxh: float,
    grading: float | None = None,
):
    _require_netgen()
    from netgen.stl import STLGeometry
    from netgen.meshing import MeshingParameters

    geo = STLGeometry(str(stl_path))
    kwargs: dict[str, Any] = {"maxh": float(maxh)}
    if grading is not None:
        kwargs["grading"] = float(grading)
    mp = MeshingParameters(**kwargs)
    nmesh = geo.GenerateMesh(mp)
    return nmesh


def generate_diffusion_mesh(
    mesh_or_path: trimesh.Trimesh | str | Path,
    *,
    target_elements: int = 2000,
    maxh: float | None = None,
    grading: float | None = None,
    units: str = "mm",
    cache_dir: str | Path | None = None,
    force_regen: bool = False,
) -> DiffusionMesh:
    """Generate a coarse tet diffusion mesh from an STL triangle mesh file using Netgen.

    Exports in-memory ``trimesh.Trimesh`` inputs to a temporary STL mesh file when needed.
    Estimates ``maxh`` from bbox volume and ``target_elements`` when ``maxh`` is
    omitted. Optionally reads/writes npz+json cache artifacts under ``cache_dir``.

    Fresh results retain ``netgen_mesh`` for NGSolve. Cache hits reload arrays
    only and set ``netgen_mesh=None`` — regenerate without cache (or with
    ``force_regen=True``) before calling :func:`solve_diffusion`.

    Parameters
    ----------
    mesh_or_path:
        STL triangle mesh file path or loaded surface mesh.
    target_elements:
        Soft element-count target for ``maxh`` estimation.
    maxh:
        Netgen maximum edge length (mesh units); estimated when ``None``.
    grading:
        Optional Netgen grading parameter.
    units:
        Length unit label stored in metadata.
    cache_dir:
        Optional directory for npz/json cache files.
    force_regen:
        Ignore existing cache and remesh.

    Returns
    -------
    DiffusionMesh

    Raises
    ------
    ImportError
        When the ``fem`` extra (Netgen) is not installed.
    RuntimeError
        When meshing produces too few tets.

    Notebook / protocol:
        M4A coarse tet mesh; requires ``pip install '.[fem]'``.
    """
    netgen = _require_netgen()
    stl_hash, geometry_id = _stl_hash(mesh_or_path)

    temporary_stl = None

    try:
        if isinstance(mesh_or_path, (str, Path)):
            stl_path = Path(mesh_or_path)
            surface = trimesh.load(stl_path, force="mesh")

        else:
            surface = mesh_or_path

            if cache_dir is not None:
                cache_dir = Path(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)

                stl_path = cache_dir / f"_tmp_{stl_hash[:16]}.stl"
                geometry_id = str(stl_path)

            else:
                tmpdir = tempfile.TemporaryDirectory(prefix="diffusion_mesh_")
                temporary_stl = tmpdir

                stl_path = Path(tmpdir.name) / f"_tmp_{stl_hash[:16]}.stl"
                geometry_id = f"in_memory:{stl_hash[:16]}"

            surface.export(stl_path)

        bounds = np.asarray(surface.bounds, dtype=float)

        if maxh is None:
            maxh = _estimate_maxh(bounds, target_elements)

        cache_keys = {
            "stl_hash": stl_hash,
            "geometry_id": geometry_id,
            "units": units,
            "meshing_method": "netgen",
            "netgen_version": getattr(netgen, "__version__", None),
            "target_elements": int(target_elements),
            "maxh": float(maxh),
            "grading": None if grading is None else float(grading),
        }

        if cache_dir is not None and not force_regen:
            cache_dir = Path(cache_dir)

            key_hash = hashlib.sha256(
                json.dumps(cache_keys, sort_keys=True).encode()
            ).hexdigest()[:16]

            npz_path = cache_dir / f"diffusion_mesh_{key_hash}.npz"
            meta_path = cache_dir / f"diffusion_mesh_{key_hash}_meta.json"

            if npz_path.is_file() and meta_path.is_file():
                data = np.load(npz_path)
                meta = json.loads(meta_path.read_text())

                return DiffusionMesh(
                    nodes=data["nodes"],
                    tets=data["tets"],
                    centroids=data["centroids"],
                    volumes=data["volumes"],
                    metadata=DiffusionMeshMetadata(
                        **{
                            k: meta[k]
                            for k in DiffusionMeshMetadata.__dataclass_fields__
                            if k in meta
                        }
                    ),
                    netgen_mesh=None,
                )

        nmesh = _mesh_stl_to_tet_netgen(
            stl_path,
            maxh=maxh,
            grading=grading,
        )

        nodes, tets = _extract_nodes_tets(nmesh)

        volumes = _tet_volumes(nodes, tets)
        centroids = nodes[tets].mean(axis=1)

        if not (200 <= len(tets) <= 50_000):
            if len(tets) < 10:
                raise RuntimeError(
                    f"Diffusion mesh too coarse: n_tets={len(tets)}"
                )

        metadata = DiffusionMeshMetadata(
            stl_hash=stl_hash,
            geometry_id=geometry_id,
            units=units,
            meshing_method="netgen",
            netgen_version=getattr(netgen, "__version__", None),
            num_elements=int(len(tets)),
            num_nodes=int(len(nodes)),
            target_elements=int(target_elements),
            maxh=float(maxh),
            grading=None if grading is None else float(grading),
            is_authoritative_geometry=False,
            resolution_basis="diffusion_length_scale",
            cache_keys=cache_keys,
        )

        diff_mesh = DiffusionMesh(
            nodes=nodes,
            tets=tets,
            centroids=centroids,
            volumes=volumes,
            metadata=metadata,
            netgen_mesh=nmesh,
        )

        if cache_dir is not None:
            cache_dir = Path(cache_dir)

            cache_dir.mkdir(parents=True, exist_ok=True)

            key_hash = hashlib.sha256(
                json.dumps(cache_keys, sort_keys=True).encode()
            ).hexdigest()[:16]

            diff_mesh.save_npz(
                cache_dir / f"diffusion_mesh_{key_hash}.npz"
            )

        return diff_mesh

    finally:
        if temporary_stl is not None:
            temporary_stl.cleanup()
