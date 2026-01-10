"""
Track Splitting for multi-radar fusion.
Detects when tracks from different radars have diverged and should be split.
"""

import numpy as np


def check_and_split_track(gid, cluster, id_manager, filters, global_tid_last_ts,
                           split_distance_threshold, split_velocity_threshold,
                           ekf_update_single_fn, ekf_update_single_with_gid_fn,
                           ekf_fuse_cluster_with_gid_fn, log_fn=None):
    """
    Check if a global track should be split because measurements from different
    radars have diverged (indicating two people who were merged are now separating).
    
    Args:
        gid: The global track ID
        cluster: List of measurements from different radars assigned to this track
        id_manager: TrackIDManager instance
        filters: dict of gid -> EKFTrack
        global_tid_last_ts: dict of gid -> timestamp
        split_distance_threshold: max position distance before split
        split_velocity_threshold: max velocity difference before split
        ekf_update_single_fn: function to create new track from single measurement
        ekf_update_single_with_gid_fn: function to update track with known GID
        ekf_fuse_cluster_with_gid_fn: function to fuse cluster with known GID
        log_fn: logging function
        
    Returns:
        None if no split needed, or list of fused_tracks if split occurred
    """
    if len(cluster) < 2:
        return None
    
    # Group measurements by radar
    by_radar = {}
    for t in cluster:
        by_radar.setdefault(t['radar_name'], []).append(t)
    
    if len(by_radar) < 2:
        # All from same radar - can't split
        return None
    
    # Get representative position for each radar (use highest confidence if multiple)
    radar_positions = {}
    radar_velocities = {}
    radar_tracks = {}
    
    for radar_name, tracks in by_radar.items():
        best = max(tracks, key=lambda t: t.get('confidence', 0.5))
        radar_positions[radar_name] = np.array([best['posX'], best['posY'], best['posZ']])
        radar_velocities[radar_name] = np.array([
            best.get('velX', 0), best.get('velY', 0), best.get('velZ', 0)
        ])
        radar_tracks[radar_name] = best
    
    # Check if any pair of radars has diverging measurements
    radar_names = list(radar_positions.keys())
    max_pos_dist = 0
    max_vel_dist = 0
    
    for i, r1 in enumerate(radar_names):
        for r2 in radar_names[i+1:]:
            pos_dist = np.linalg.norm(radar_positions[r1] - radar_positions[r2])
            vel_dist = np.linalg.norm(radar_velocities[r1] - radar_velocities[r2])
            
            max_pos_dist = max(max_pos_dist, pos_dist)
            max_vel_dist = max(max_vel_dist, vel_dist)
    
    # Check if split is needed
    needs_split = (max_pos_dist > split_distance_threshold or 
                   max_vel_dist > split_velocity_threshold)
    
    if not needs_split:
        return None
    
    # SPLIT THE TRACK
    if log_fn:
        log_fn(f'SPLIT detected for GID {gid}: pos_dist={max_pos_dist:.2f}m, vel_dist={max_vel_dist:.2f}m/s')
    
    # Strategy: Keep the original GID for the radar with the oldest association,
    # create new GIDs for other radars
    fused_results = []
    
    # Find which radar had the original association (check association order)
    original_radar = None
    for radar_name in radar_names:
        key = (radar_name, radar_tracks[radar_name]['tid'])
        existing_gid = id_manager.get_global_tid(radar_name, radar_tracks[radar_name]['tid'])
        if existing_gid == gid:
            if original_radar is None:
                original_radar = radar_name
                break
    
    if original_radar is None:
        original_radar = radar_names[0]
    
    # Process each radar's measurements separately
    for radar_name, tracks in by_radar.items():
        if radar_name == original_radar:
            # Keep original GID for this radar
            if len(tracks) == 1:
                fused_results.append(ekf_update_single_with_gid_fn(tracks[0], gid))
            else:
                fused_results.append(ekf_fuse_cluster_with_gid_fn(tracks, gid))
        else:
            # Create new GID for this radar - break the association
            for t in tracks:
                # Remove old association
                id_manager.remove_mapping(t['radar_name'], t['tid'])
                # Create as new track
                fused_results.append(ekf_update_single_fn(t))
    
    return fused_results
