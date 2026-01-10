"""
Track Fusion Package for Multi-Radar Systems

This package provides EKF-based track fusion for dual or multi-radar systems.
It fuses tracks into persistent global tracks using a Constant Acceleration (CA) model.

Usage:
    from library.track_fusion_pkg import TrackFusion, TrackFusionVisualizer
    
    fusion = TrackFusion(
        TRACK_FUSION_CFG={...},
        RADAR_CFG_LIST=[...]
    )
    
    fused_tracks = fusion.fuse_tracks(radar_frames)

Modules:
    - ekf: Extended Kalman Filter implementation (CA model)
    - id_manager: Track ID management and lifecycle
    - association: Cross-radar association and late fusion
    - splitting: Track split detection
    - fusion: Main TrackFusion class
    - visualizer: Track visualization helper
"""

from .fusion import TrackFusion
from .visualizer import TrackFusionVisualizer
from .ekf import EKFTrack
from .id_manager import TrackIDManager

__all__ = [
    'TrackFusion',
    'TrackFusionVisualizer',
    'EKFTrack',
    'TrackIDManager',
]
