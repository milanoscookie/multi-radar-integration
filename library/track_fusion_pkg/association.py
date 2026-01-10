"""
Track Association for multi-radar fusion.
Handles cross-radar association, late fusion, and conflict zone detection.
"""

import numpy as np


def detect_conflict_zones(existing_track_positions, min_track_separation):
    """
    Detect areas where multiple existing global tracks are close together.
    In these zones, we should NOT allow new cross-radar fusion to prevent
    accidentally merging distinct people.
    
    Args:
        existing_track_positions: dict of gid -> np.array([x, y, z])
        min_track_separation: minimum separation distance (conflict zone is 3x this)
        
    Returns:
        set of (gid1, gid2) pairs that are in conflict
    """
    conflicts = set()
    gids = list(existing_track_positions.keys())
    
    for i, gid1 in enumerate(gids):
        for gid2 in gids[i+1:]:
            pos1 = existing_track_positions[gid1]
            pos2 = existing_track_positions[gid2]
            dist = np.linalg.norm(pos1 - pos2)
            
            if dist < min_track_separation * 3:  # Conflict zone is 3x min separation
                conflicts.add((min(gid1, gid2), max(gid1, gid2)))
    
    return conflicts


def associate_new_measurements(all_tracks, new_indices, existing_positions, conflict_zone,
                                min_track_separation, distance_threshold, velocity_threshold,
                                debug=False, log_fn=None):
    """
    Associate new measurements, with awareness of conflict zones.
    
    Rules:
    1. If a new measurement is near an existing track that's in a conflict zone,
       DON'T fuse it cross-radar - let each radar maintain its own track
    2. Only allow cross-radar fusion in "clear" areas
    
    Args:
        all_tracks: list of all track dicts
        new_indices: list of indices into all_tracks that are new (no global TID yet)
        existing_positions: dict of gid -> np.array([x, y, z])
        conflict_zone: set of (gid1, gid2) pairs in conflict
        min_track_separation: separation threshold
        distance_threshold: max distance for cross-radar association
        velocity_threshold: max velocity diff for cross-radar association
        debug: enable debug logging
        log_fn: logging function
        
    Returns:
        list of clusters, where each cluster is a list of indices
    """
    if not new_indices:
        return []

    # Check which new measurements are in conflict zones
    in_conflict = set()
    for idx in new_indices:
        t = all_tracks[idx]
        pos = np.array([t['posX'], t['posY'], t['posZ']])
        
        for gid, gpos in existing_positions.items():
            dist = np.linalg.norm(pos - gpos)
            if dist < min_track_separation * 2:
                # Check if this gid is involved in any conflict
                for g1, g2 in conflict_zone:
                    if gid == g1 or gid == g2:
                        in_conflict.add(idx)
                        break

    # Measurements in conflict zones become individual tracks (no cross-radar fusion)
    clusters = []
    for idx in new_indices:
        if idx in in_conflict:
            clusters.append([idx])

    # For non-conflict measurements, apply normal cross-radar association
    safe_indices = [idx for idx in new_indices if idx not in in_conflict]
    
    if not safe_indices:
        return clusters

    # Group by radar
    by_radar = {}
    for idx in safe_indices:
        r = all_tracks[idx]['radar_name']
        by_radar.setdefault(r, []).append(idx)

    # Cross-radar association with position + velocity consistency
    safe_clusters = cross_radar_associate(
        all_tracks, by_radar, distance_threshold, velocity_threshold, debug, log_fn
    )
    clusters.extend(safe_clusters)

    return clusters


def cross_radar_associate(all_tracks, by_radar, distance_threshold, velocity_threshold,
                           debug=False, log_fn=None):
    """
    Associate unassociated measurements across different radars.
    Only creates clusters from measurements belonging to DIFFERENT radars.
    Uses both position and velocity consistency.
    
    Args:
        all_tracks: list of all track dicts
        by_radar: dict of radar_name -> list of indices
        distance_threshold: max position distance for association
        velocity_threshold: max velocity difference for association
        debug: enable debug logging
        log_fn: logging function
        
    Returns:
        list of clusters, where each cluster is a list of indices
    """
    radar_names = list(by_radar.keys())
    if len(radar_names) < 2:
        # No cross-radar fusion possible, return each as individual
        return [[idx] for indices in by_radar.values() for idx in indices]

    # Collect all unassociated indices
    all_unassoc = []
    for indices in by_radar.values():
        all_unassoc.extend(indices)

    if not all_unassoc:
        return []

    n = len(all_unassoc)
    if n == 1:
        return [all_unassoc]

    # Build affinity matrix: can only link if DIFFERENT radar AND consistent pos+vel
    can_merge = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            idx_i = all_unassoc[i]
            idx_j = all_unassoc[j]
            t_i = all_tracks[idx_i]
            t_j = all_tracks[idx_j]

            # CRITICAL: Only allow merge if from DIFFERENT radars
            if t_i['radar_name'] == t_j['radar_name']:
                continue

            # Check position consistency
            pos_i = np.array([t_i['posX'], t_i['posY'], t_i['posZ']])
            pos_j = np.array([t_j['posX'], t_j['posY'], t_j['posZ']])
            pos_dist = np.linalg.norm(pos_i - pos_j)

            if pos_dist > distance_threshold:
                if debug and log_fn:
                    log_fn(f'    Reject merge {t_i["radar_name"]}:{t_i["tid"]} <-> {t_j["radar_name"]}:{t_j["tid"]}: pos_dist={pos_dist:.3f}m > {distance_threshold}m')
                continue

            # Check velocity consistency
            vel_i = np.array([t_i.get('velX', 0), t_i.get('velY', 0), t_i.get('velZ', 0)])
            vel_j = np.array([t_j.get('velX', 0), t_j.get('velY', 0), t_j.get('velZ', 0)])
            vel_dist = np.linalg.norm(vel_i - vel_j)

            if vel_dist > velocity_threshold:
                if debug and log_fn:
                    log_fn(f'    Reject merge {t_i["radar_name"]}:{t_i["tid"]} <-> {t_j["radar_name"]}:{t_j["tid"]}: vel_dist={vel_dist:.3f}m/s > {velocity_threshold}m/s')
                continue

            if debug and log_fn:
                log_fn(f'    Allow merge {t_i["radar_name"]}:{t_i["tid"]} <-> {t_j["radar_name"]}:{t_j["tid"]}: pos={pos_dist:.3f}m, vel={vel_dist:.3f}m/s')
            can_merge[i, j] = True
            can_merge[j, i] = True

    # Connected components on the affinity graph (BFS)
    visited = set()
    clusters = []

    for start in range(n):
        if start in visited:
            continue

        cluster = []
        stack = [start]

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            cluster.append(all_unassoc[cur])  # Store original index

            for nb in range(n):
                if nb not in visited and can_merge[cur, nb]:
                    stack.append(nb)

        # Validate cluster: should contain tracks from at least 2 radars if merged
        if len(cluster) > 1:
            radars_in_cluster = set(all_tracks[idx]['radar_name'] for idx in cluster)
            if len(radars_in_cluster) >= 2:
                clusters.append(cluster)
            else:
                # Split back to individuals (shouldn't happen due to logic above)
                for idx in cluster:
                    clusters.append([idx])
        else:
            clusters.append(cluster)

    return clusters


def attempt_late_fusion(all_tracks, established_measurements, filters, id_manager,
                         global_tid_last_ts, late_fusion_distance, late_fusion_velocity,
                         debug=False, log_fn=None):
    """
    LATE FUSION: Merge established tracks from DIFFERENT radars if they are close.
    
    This fixes the async detection problem:
    - Radar_A detects person first -> gets GID 1
    - Radar_B detects same person 100ms later -> gets GID 2
    - Now they're locked to separate GIDs forever
    
    Solution: If GID 1 and GID 2 are consistently close (same person), merge them.
    
    Args:
        all_tracks: list of all track dicts
        established_measurements: list of (idx, gid) tuples
        filters: dict of gid -> EKFTrack
        id_manager: TrackIDManager instance
        global_tid_last_ts: dict of gid -> last measurement timestamp
        late_fusion_distance: max position distance for late fusion
        late_fusion_velocity: max velocity difference for late fusion
        debug: enable debug logging
        log_fn: logging function
        
    Returns:
        Updated established_measurements list with merged GIDs
    """
    # Group measurements by their current global_tid
    gid_to_measurements = {}
    for idx, gid in established_measurements:
        gid_to_measurements.setdefault(gid, []).append((idx, all_tracks[idx]))
    
    if len(gid_to_measurements) < 2:
        return established_measurements
    
    # Get current EKF positions for each GID
    gid_positions = {}
    gid_velocities = {}
    gid_radars = {}
    
    for gid, measurements in gid_to_measurements.items():
        if gid in filters:
            x = filters[gid].x.squeeze()
            gid_positions[gid] = np.array([x[0], x[1], x[2]])
            gid_velocities[gid] = np.array([x[3], x[4], x[5]])
        else:
            # Use measurement position
            t = measurements[0][1]
            gid_positions[gid] = np.array([t['posX'], t['posY'], t['posZ']])
            gid_velocities[gid] = np.array([t.get('velX', 0), t.get('velY', 0), t.get('velZ', 0)])
        
        # Track which radars contribute to this GID
        gid_radars[gid] = set(t['radar_name'] for _, t in measurements)
    
    # Find pairs of GIDs that should be merged (different radars, close position/velocity)
    gids = list(gid_to_measurements.keys())
    merge_pairs = []
    
    for i, gid1 in enumerate(gids):
        for gid2 in gids[i+1:]:
            # Only merge if from DIFFERENT radars
            if gid_radars[gid1] & gid_radars[gid2]:
                # Same radar appears in both - don't merge
                continue
            
            pos_dist = np.linalg.norm(gid_positions[gid1] - gid_positions[gid2])
            vel_dist = np.linalg.norm(gid_velocities[gid1] - gid_velocities[gid2])
            
            if pos_dist <= late_fusion_distance and vel_dist <= late_fusion_velocity:
                merge_pairs.append((gid1, gid2, pos_dist, vel_dist))
                if debug and log_fn:
                    log_fn(f'  LATE FUSION candidate: GID {gid1} + GID {gid2} '
                           f'(pos={pos_dist:.3f}m, vel={vel_dist:.3f}m/s)')
    
    if not merge_pairs:
        return established_measurements
    
    # Process merges: keep the lower GID (older track), redirect higher GID
    for gid1, gid2, pos_dist, vel_dist in merge_pairs:
        keep_gid = min(gid1, gid2)
        remove_gid = max(gid1, gid2)
        
        if log_fn:
            log_fn(f'LATE FUSION: Merging GID {remove_gid} into GID {keep_gid} '
                   f'(pos={pos_dist:.3f}m, vel={vel_dist:.3f}m/s)')
        
        # Merge EKF state: keep the filter with lower covariance trace (more confident)
        if remove_gid in filters and keep_gid in filters:
            keep_trace = np.trace(filters[keep_gid].P)
            remove_trace = np.trace(filters[remove_gid].P)
            
            if remove_trace < keep_trace:
                # The removed filter is more confident - use its state
                filters[keep_gid].x = filters[remove_gid].x.copy()
                filters[keep_gid].P = filters[remove_gid].P.copy()
        
        # Use id_manager to handle the merge
        keys_updated = id_manager.merge_tids(keep_gid, remove_gid, filters, global_tid_last_ts)
        
        if debug and log_fn:
            for key in keys_updated:
                log_fn(f'    Remapped {key} -> GID {keep_gid}')
    
    # Rebuild established_measurements with updated GIDs
    new_established = []
    for idx, old_gid in established_measurements:
        track = all_tracks[idx]
        key = (track['radar_name'], track['tid'])
        new_gid = id_manager.get_global_tid(track['radar_name'], track['tid'])
        if new_gid is None:
            new_gid = old_gid
        new_established.append((idx, new_gid))
    
    return new_established
