"""
Gaussian Splat Merger

Combines multiple Gaussian splat .ply files into a single file,
applying position/rotation/scale transforms to each splat's points.

PLY format for Gaussian splats contains:
- position: x, y, z
- normal: nx, ny, nz
- SH coefficients: f_dc_0..2, f_rest_0..44 (spherical harmonics for color)
- opacity: opacity
- scale: scale_0, scale_1, scale_2
- rotation: rot_0, rot_1, rot_2, rot_3 (quaternion)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SplatData:
    """Container for Gaussian splat data"""
    positions: np.ndarray      # (N, 3) xyz positions
    normals: np.ndarray        # (N, 3) normals
    sh_dc: np.ndarray          # (N, 3) DC spherical harmonics (base color)
    sh_rest: np.ndarray        # (N, 45) rest of SH coefficients
    opacities: np.ndarray      # (N,) opacity values
    scales: np.ndarray         # (N, 3) scale values
    rotations: np.ndarray      # (N, 4) quaternion rotations

    @property
    def num_splats(self) -> int:
        return len(self.positions)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (w, x, y, z format).
    q1 and q2 can be single quaternions (4,) or arrays of quaternions (N, 4).
    """
    # Handle both single quaternion and array of quaternions
    if q1.ndim == 1:
        q1 = q1.reshape(1, 4)
    if q2.ndim == 1:
        q2 = q2.reshape(1, 4)

    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.stack([w, x, y, z], axis=-1)


def euler_to_quaternion(euler: List[float]) -> np.ndarray:
    """
    Convert Euler angles (degrees) to quaternion (w, x, y, z).
    Euler angles are [rx, ry, rz] in degrees.
    """
    rx, ry, rz = np.radians(euler)

    # Calculate quaternion components
    cx, sx = np.cos(rx/2), np.sin(rx/2)
    cy, sy = np.cos(ry/2), np.sin(ry/2)
    cz, sz = np.cos(rz/2), np.sin(rz/2)

    # ZYX rotation order
    w = cx*cy*cz + sx*sy*sz
    x = sx*cy*cz - cx*sy*sz
    y = cx*sy*cz + sx*cy*sz
    z = cx*cy*sz - sx*sy*cz

    return np.array([w, x, y, z])


def rotate_points_by_quaternion(points: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate points by quaternion.
    points: (N, 3) array of xyz positions
    q: (4,) quaternion (w, x, y, z)
    """
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    # Build rotation matrix from quaternion
    rot_matrix = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

    return points @ rot_matrix.T


def load_ply(path: str) -> SplatData:
    """
    Load a Gaussian splat PLY file.
    Returns SplatData with all splat properties.
    """
    from plyfile import PlyData

    logger.info(f"Loading PLY: {path}")
    plydata = PlyData.read(path)
    vertex = plydata['vertex']

    # Extract positions
    positions = np.stack([
        vertex['x'],
        vertex['y'],
        vertex['z']
    ], axis=-1).astype(np.float32)

    # Extract normals (may not exist in all files)
    try:
        normals = np.stack([
            vertex['nx'],
            vertex['ny'],
            vertex['nz']
        ], axis=-1).astype(np.float32)
    except ValueError:
        normals = np.zeros_like(positions)

    # Extract SH DC coefficients (base color)
    sh_dc = np.stack([
        vertex['f_dc_0'],
        vertex['f_dc_1'],
        vertex['f_dc_2']
    ], axis=-1).astype(np.float32)

    # Extract SH rest coefficients
    sh_rest_names = [f'f_rest_{i}' for i in range(45)]
    sh_rest_list = []
    for name in sh_rest_names:
        try:
            sh_rest_list.append(vertex[name])
        except ValueError:
            sh_rest_list.append(np.zeros(len(positions)))
    sh_rest = np.stack(sh_rest_list, axis=-1).astype(np.float32)

    # Extract opacity
    opacities = vertex['opacity'].astype(np.float32)

    # Extract scales
    scales = np.stack([
        vertex['scale_0'],
        vertex['scale_1'],
        vertex['scale_2']
    ], axis=-1).astype(np.float32)

    # Extract rotations (quaternion)
    rotations = np.stack([
        vertex['rot_0'],
        vertex['rot_1'],
        vertex['rot_2'],
        vertex['rot_3']
    ], axis=-1).astype(np.float32)

    logger.info(f"Loaded {len(positions)} splats from {path}")

    return SplatData(
        positions=positions,
        normals=normals,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        opacities=opacities,
        scales=scales,
        rotations=rotations
    )


def transform_splat(
    splat: SplatData,
    position: List[float] = None,
    rotation: List[float] = None,
    scale: List[float] = None
) -> SplatData:
    """
    Apply transform to splat data.

    Args:
        splat: Input SplatData
        position: [x, y, z] translation
        rotation: [rx, ry, rz] Euler angles in degrees
        scale: [sx, sy, sz] scale factors

    Returns:
        New SplatData with transforms applied
    """
    positions = splat.positions.copy()
    rotations = splat.rotations.copy()
    scales = splat.scales.copy()

    # Apply scale to positions and splat scales
    if scale is not None:
        scale_arr = np.array(scale, dtype=np.float32)
        positions = positions * scale_arr
        # Scale affects the splat sizes too (in log space for scales)
        scales = scales + np.log(scale_arr)

    # Apply rotation to positions and splat rotations
    if rotation is not None and any(r != 0 for r in rotation):
        q = euler_to_quaternion(rotation)
        positions = rotate_points_by_quaternion(positions, q)
        # Rotate each splat's orientation quaternion
        rotations = quaternion_multiply(q.reshape(1, 4), rotations)
        # Normalize quaternions
        rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)

    # Apply translation
    if position is not None:
        positions = positions + np.array(position, dtype=np.float32)

    return SplatData(
        positions=positions,
        normals=splat.normals.copy(),
        sh_dc=splat.sh_dc.copy(),
        sh_rest=splat.sh_rest.copy(),
        opacities=splat.opacities.copy(),
        scales=scales,
        rotations=rotations
    )


def merge_splats(splats: List[SplatData]) -> SplatData:
    """
    Merge multiple SplatData objects into one.
    """
    if not splats:
        raise ValueError("No splats to merge")

    if len(splats) == 1:
        return splats[0]

    return SplatData(
        positions=np.concatenate([s.positions for s in splats], axis=0),
        normals=np.concatenate([s.normals for s in splats], axis=0),
        sh_dc=np.concatenate([s.sh_dc for s in splats], axis=0),
        sh_rest=np.concatenate([s.sh_rest for s in splats], axis=0),
        opacities=np.concatenate([s.opacities for s in splats], axis=0),
        scales=np.concatenate([s.scales for s in splats], axis=0),
        rotations=np.concatenate([s.rotations for s in splats], axis=0),
    )


def save_ply(splat: SplatData, path: str):
    """
    Save SplatData to a PLY file.
    """
    from plyfile import PlyData, PlyElement

    num_splats = splat.num_splats

    # Build vertex dtype
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]
    for i in range(45):
        dtype.append((f'f_rest_{i}', 'f4'))
    dtype.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ])

    # Create structured array
    vertices = np.zeros(num_splats, dtype=dtype)

    # Fill in data
    vertices['x'] = splat.positions[:, 0]
    vertices['y'] = splat.positions[:, 1]
    vertices['z'] = splat.positions[:, 2]

    vertices['nx'] = splat.normals[:, 0]
    vertices['ny'] = splat.normals[:, 1]
    vertices['nz'] = splat.normals[:, 2]

    vertices['f_dc_0'] = splat.sh_dc[:, 0]
    vertices['f_dc_1'] = splat.sh_dc[:, 1]
    vertices['f_dc_2'] = splat.sh_dc[:, 2]

    for i in range(45):
        vertices[f'f_rest_{i}'] = splat.sh_rest[:, i]

    vertices['opacity'] = splat.opacities

    vertices['scale_0'] = splat.scales[:, 0]
    vertices['scale_1'] = splat.scales[:, 1]
    vertices['scale_2'] = splat.scales[:, 2]

    vertices['rot_0'] = splat.rotations[:, 0]
    vertices['rot_1'] = splat.rotations[:, 1]
    vertices['rot_2'] = splat.rotations[:, 2]
    vertices['rot_3'] = splat.rotations[:, 3]

    # Create PLY element and save
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(path)

    logger.info(f"Saved {num_splats} splats to {path}")


def merge_scene_splats(
    splat_configs: List[Dict],
    output_path: str
) -> str:
    """
    Merge multiple splats with transforms into a single file.

    Args:
        splat_configs: List of dicts with:
            - path: Path to .ply file
            - transform: {position: [x,y,z], rotation: [rx,ry,rz], scale: [sx,sy,sz]}
        output_path: Where to save merged .ply

    Returns:
        Path to merged file
    """
    logger.info(f"Merging {len(splat_configs)} splats...")

    transformed_splats = []

    for config in splat_configs:
        splat_path = config['path']
        transform = config.get('transform', {})

        # Load splat
        splat = load_ply(splat_path)

        # Apply transform
        transformed = transform_splat(
            splat,
            position=transform.get('position'),
            rotation=transform.get('rotation'),
            scale=transform.get('scale')
        )

        transformed_splats.append(transformed)
        logger.info(f"  Added {splat.num_splats} splats from {splat_path}")

    # Merge all
    merged = merge_splats(transformed_splats)

    # Save
    save_ply(merged, output_path)

    logger.info(f"Merged scene saved: {output_path} ({merged.num_splats} total splats)")
    return output_path
