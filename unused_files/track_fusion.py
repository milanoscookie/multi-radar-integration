"""
Track Fusion Module for Dual (or Multi) Radar System
Fuses tracks into persistent global tracks using an EKF (CA model).
State: [px, py, pz, vx, vy, vz, ax, ay, az]^T
Measurements: position, velocity, and acceleration

================================================================================
EKF TUNING GUIDE
================================================================================

CONFIGURATION PARAMETERS (set in TRACK_FUSION_CFG dict):
---------------------------------------------------------

1. sigma_j (default: 1.0) - Process noise: Jerk standard deviation [m/s^3]
   
   This controls how much the filter expects acceleration to change between updates.
   
   - INCREASE (e.g., 2.0-5.0) if:
     * Targets are highly maneuvering (sudden starts/stops, direction changes)
     * Tracks are lagging behind fast-moving targets
     * You see systematic position errors during maneuvers
   
   - DECREASE (e.g., 0.1-0.5) if:
     * Targets move smoothly with gradual acceleration changes
     * Tracks are too jittery/noisy
     * You want smoother trajectory estimates

2. meas_sigma_pos (default: 0.25) - Position measurement noise [m]
   
   This is the expected standard deviation of position measurements from your radar.
   
   - Set to your radar's position accuracy specification
   - INCREASE if radar has poor position accuracy or environment is noisy
   - DECREASE if radar has high position accuracy (will trust measurements more)

3. meas_sigma_vel (default: 0.50) - Velocity measurement noise [m/s]
   
   This is the expected standard deviation of velocity measurements.
   
   - Set based on your radar's Doppler/velocity measurement accuracy
   - Typically 2-3x the position noise for radar systems
   - INCREASE if velocity measurements are noisy or unreliable

4. meas_sigma_acc (default: 1.0) - Acceleration measurement noise [m/s^2]
   
   This is the expected standard deviation of acceleration measurements.
   
   - Radar-derived acceleration is often less accurate than position/velocity
   - INCREASE if acceleration measurements are computed (not directly measured)
   - Set higher (2.0-5.0) if acceleration is derived from velocity differencing

5. mahalanobis_gate (default: 22.0) - Gating threshold for outlier rejection
   
   This is the chi-squared threshold for rejecting outlier measurements.
   For a 9-dimensional measurement (pos+vel+acc), chi2(9) at 99% = ~21.7
   
   - INCREASE (e.g., 25-30) to accept more measurements (looser gating)
   - DECREASE (e.g., 15-18) to reject more outliers (stricter gating)
   - Set to chi2(dim) at desired confidence: chi2(9)@95%=16.9, chi2(6)@99%=16.8

6. distance_threshold (default: 0.5) - Track association distance [m]
   
   Maximum Euclidean distance to consider two tracks as the same target.
   
   - INCREASE if radars have large position offsets or targets are sparse
   - DECREASE if targets are close together to avoid false associations

7. velocity_threshold (default: 1.0) - Velocity consistency threshold [m/s]
   
   Maximum velocity difference to consider two tracks as the same target.
   This prevents merging tracks of two people walking in different directions
   even if they are spatially close.
   
   - INCREASE if velocity measurements are noisy or unreliable
   - DECREASE to be stricter about motion consistency (e.g., 0.5 for slow walkers)
   - Set based on expected velocity measurement accuracy

8. tid_timeout (default: 2.0) - Track ID timeout [seconds]
   
   How long to keep a track alive after last detection before deleting it.
   
   - INCREASE for intermittent detections or slow-moving targets
   - DECREASE for fast update rates or to quickly drop lost tracks

9. min_track_separation (default: 0.3) - Minimum separation for conflict zones [m]
    
    When existing global tracks are closer than 3x this value (conflict zone),
    new measurements in that area will NOT be cross-radar fused. This prevents
    accidental merging when two people are close together.
    
    - INCREASE to create larger "no fusion" zones around close tracks
    - DECREASE if fusion is being prevented in valid scenarios
    - Note: Track identity is locked once established (via radar_to_global_tid mapping)

10. split_distance_threshold (default: 0.8) - Track split position threshold [m]
    
    If measurements from different radars assigned to the SAME global track
    are further apart than this distance, the track will be SPLIT into separate
    tracks. This recovers from incorrect merges when two people separate.
    
    - INCREASE if tracks are splitting too aggressively (normal movement causes splits)
    - DECREASE if merged tracks aren't splitting when people walk apart
    - Should be larger than distance_threshold to avoid oscillation

11. split_velocity_threshold (default: 0.8) - Track split velocity threshold [m/s]
    
    If measurements from different radars assigned to the SAME global track
    have velocity differences larger than this, the track will be SPLIT.
    This helps detect when merged people start moving in different directions.
    
    - INCREASE if tracks are splitting due to noisy velocity measurements
    - DECREASE to be more sensitive to diverging motion

12. enable_late_fusion (default: True) - Enable late fusion for async radar detection
    
    When radars detect the same person at slightly different times, they get
    assigned separate global TIDs. Late fusion detects this and merges them.
    
    - Set to True (default) to automatically merge tracks that should be together
    - Set to False to disable (if you want strict per-radar tracking)

13. late_fusion_distance (default: 0.5) - Late fusion position threshold [m]
    
    Maximum distance between two established tracks from DIFFERENT radars
    for them to be merged via late fusion.
    
    - INCREASE if radars have larger position offsets
    - DECREASE to be stricter about which tracks get merged

14. late_fusion_velocity (default: 1.0) - Late fusion velocity threshold [m/s]
    
    Maximum velocity difference between two established tracks for late fusion.
    
    - INCREASE if velocity measurements are noisy
    - DECREASE to require more consistent motion for fusion

EXAMPLE CONFIGURATION:
----------------------
fusion_cfg = {
    'TRACK_FUSION_CFG': {
        'distance_threshold': 0.5,       # meters - for initial cross-radar fusion
        'velocity_threshold': 1.0,       # m/s - velocity consistency for fusion
        'min_track_separation': 0.3,     # meters - conflict zone threshold
        'split_distance_threshold': 0.8, # meters - trigger split when measurements diverge
        'split_velocity_threshold': 0.8, # m/s - trigger split on velocity divergence
        'enable_late_fusion': True,      # merge tracks that were created separately
        'late_fusion_distance': 0.5,     # meters - max distance for late fusion
        'late_fusion_velocity': 1.0,     # m/s - max velocity diff for late fusion
        'sigma_j': 1.0,                  # m/s^3 - increase for maneuvering targets
        'meas_sigma_pos': 0.25,          # m - match to radar spec
        'meas_sigma_vel': 0.50,          # m/s - match to radar spec
        'meas_sigma_acc': 1.0,           # m/s^2 - usually higher uncertainty
        'use_mahalanobis_gating': True,
        'mahalanobis_gate': 22.0,        # chi2(9) @ 99%
        'tid_timeout': 2.0,              # seconds
    },
    'RADAR_CFG_LIST': [...]
}

TUNING TIPS:
------------
- Start with defaults, observe behavior, then adjust one parameter at a time
- If tracks are too smooth/laggy: increase sigma_j
- If tracks are too noisy/jittery: decrease sigma_j or increase meas_sigma_*
- If valid measurements are being rejected: increase mahalanobis_gate
- If outliers are corrupting tracks: decrease mahalanobis_gate
- Log P_trace from fused tracks to monitor filter convergence

================================================================================
"""

import numpy as np
import time


def _as_col(x):
    x = np.asarray(x, dtype=float)
    return x.reshape((-1, 1))


class EKFTrack:
    """
    EKF for a constant-acceleration (CA) model in 3D.
    State: [px, py, pz, vx, vy, vz, ax, ay, az]^T (9 dimensions)
    """
    def __init__(self, x0, P0, sigma_j=1.0):
        self.x = _as_col(x0)              # (9,1)
        self.P = np.array(P0, dtype=float)  # (9,9)
        self.sigma_j = float(sigma_j)     # jerk std deviation (m/s^3)

    def predict(self, dt):
        dt = float(max(dt, 0.0))
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        
        # State transition for CA model
        # p' = p + v*dt + 0.5*a*dt^2
        # v' = v + a*dt
        # a' = a
        F = np.eye(9)
        F[0, 3] = dt;  F[0, 6] = 0.5 * dt2  # px
        F[1, 4] = dt;  F[1, 7] = 0.5 * dt2  # py
        F[2, 5] = dt;  F[2, 8] = 0.5 * dt2  # pz
        F[3, 6] = dt  # vx
        F[4, 7] = dt  # vy
        F[5, 8] = dt  # vz

        # Process noise: white-noise jerk model
        q = self.sigma_j ** 2
        Q1 = np.array([
            [dt5/20, dt4/8,  dt3/6],
            [dt4/8,  dt3/3,  dt2/2],
            [dt3/6,  dt2/2,  dt    ]
        ]) * q
        
        Q = np.zeros((9, 9))
        Q[np.ix_([0, 3, 6], [0, 3, 6])] = Q1  # x-axis
        Q[np.ix_([1, 4, 7], [1, 4, 7])] = Q1  # y-axis
        Q[np.ix_([2, 5, 8], [2, 5, 8])] = Q1  # z-axis

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, R, meas_type='pos_vel_acc'):
        """
        Update with measurement.
        meas_type: 'pos' (3D), 'pos_vel' (6D), or 'pos_vel_acc' (9D)
        """
        z = _as_col(z)
        R = np.array(R, dtype=float)

        if meas_type == 'pos_vel_acc':
            H = np.eye(9)
        elif meas_type == 'pos_vel':
            H = np.zeros((6, 9))
            H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1
            H[3, 3] = 1; H[4, 4] = 1; H[5, 5] = 1
        else:  # pos only
            H = np.zeros((3, 9))
            H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1

        hx = H @ self.x
        y = z - hx
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(9)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        nis = float((y.T @ np.linalg.inv(S) @ y).squeeze())
        return nis

    def get_state(self):
        x = self.x.squeeze()
        return {
            "posX": float(x[0]), "posY": float(x[1]), "posZ": float(x[2]),
            "velX": float(x[3]), "velY": float(x[4]), "velZ": float(x[5]),
            "accX": float(x[6]), "accY": float(x[7]), "accZ": float(x[8]),
        }


class TrackFusion:
    def __init__(self, **kwargs_CFG):
        self.fusion_cfg = kwargs_CFG.get('TRACK_FUSION_CFG', {})
        self.radar_cfg_list = kwargs_CFG['RADAR_CFG_LIST']

        # Original knobs (still supported)
        self.distance_threshold = self.fusion_cfg.get('distance_threshold', 0.5)  # meters

        # Velocity consistency threshold for association
        # Two tracks only merge if velocity difference < this threshold
        self.velocity_threshold = self.fusion_cfg.get('velocity_threshold', 1.0)  # m/s

        # EKF knobs (CA model)
        self.sigma_j = self.fusion_cfg.get('sigma_j', 1.0)    # m/s^3 process jerk std
        self.meas_sigma_pos = self.fusion_cfg.get('meas_sigma_pos', 0.25)  # m
        self.meas_sigma_vel = self.fusion_cfg.get('meas_sigma_vel', 0.50)  # m/s
        self.meas_sigma_acc = self.fusion_cfg.get('meas_sigma_acc', 1.0)   # m/s^2
        self.use_mahalanobis_gating = self.fusion_cfg.get('use_mahalanobis_gating', True)
        self.mahalanobis_gate = self.fusion_cfg.get('mahalanobis_gate', 22.0)  # chi2(9) ~21.7 @ 99%

        # Track ID management (kept)
        self.next_global_tid = 1
        self.radar_to_global_tid = {}      # (radar_name, local_tid) -> global_tid
        self.global_tid_last_seen = {}     # global_tid -> time.time()
        self.tid_timeout = float(self.fusion_cfg.get('tid_timeout', 2.0))

        # Track association lock-in is implicit:
        # Once a (radar_name, local_tid) is mapped to a global_tid in radar_to_global_tid,
        # that association is LOCKED until the global track times out.
        # This prevents track hijacking when people get close.
        
        # Minimum separation distance to PREVENT fusion
        # If two existing global tracks are closer than this, DON'T fuse new measurements into them
        self.min_track_separation = float(self.fusion_cfg.get('min_track_separation', 0.3))  # meters

        # Track splitting parameters
        # If measurements from different radars assigned to the SAME global track
        # are further apart than this, trigger a split
        self.split_distance_threshold = float(self.fusion_cfg.get('split_distance_threshold', 0.8))  # meters
        self.split_velocity_threshold = float(self.fusion_cfg.get('split_velocity_threshold', 0.8))  # m/s

        # Late fusion: merge established tracks from different radars if they stay close
        # This fixes the issue where radars detect the same person at slightly different times
        self.enable_late_fusion = self.fusion_cfg.get('enable_late_fusion', True)
        self.late_fusion_distance = float(self.fusion_cfg.get('late_fusion_distance', 0.5))  # meters
        self.late_fusion_velocity = float(self.fusion_cfg.get('late_fusion_velocity', 1.0))  # m/s

        # Global EKF bank
        self.filters = {}                  # global_tid -> EKFTrack
        self.global_tid_last_ts = {}       # global_tid -> last measurement timestamp

        # Debug mode
        self.debug = self.fusion_cfg.get('debug', False)

        self._log('Track Fusion (EKF-CA) initialized with track lock-in and split detection')

    def fuse_tracks(self, radar_frames):
        """
        Main fusion method with TRACK IDENTITY PRESERVATION:
        
        Key principle: Once a local track ID is associated with a global track,
        that association is LOCKED and cannot be changed by proximity to other tracks.
        
        Algorithm:
        1. For each measurement, check if its (radar, local_tid) already has a global association
        2. If YES -> update that global track (no cross-radar fusion allowed for established tracks)
        3. If NO -> this is a NEW track, try to find cross-radar matches for initial fusion
        4. Prevent fusion when existing global tracks are too close to each other
        """
        if not radar_frames:
            return []

        self._cleanup_old_tids()

        # Flatten measurements from all radars
        all_tracks = []
        for frame in radar_frames:
            radar_name = frame['radar_name']
            ts = float(frame['timestamp'])
            for track in frame['tracks']:
                t = track.copy()
                t['radar_name'] = radar_name
                t['timestamp'] = ts
                all_tracks.append(t)

        if not all_tracks:
            return []

        # Separate measurements into ESTABLISHED (have global association) and NEW
        established_measurements = []  # [(idx, gid), ...]
        new_measurements = []          # [idx, ...]

        for idx, track in enumerate(all_tracks):
            key = (track['radar_name'], track['tid'])
            if key in self.radar_to_global_tid:
                gid = self.radar_to_global_tid[key]
                if gid in self.filters:
                    established_measurements.append((idx, gid))
                    if self.debug:
                        self._log(f'  ESTABLISHED: {key} -> GID {gid}')
                else:
                    # Filter was cleaned up, treat as new
                    new_measurements.append(idx)
                    if self.debug:
                        self._log(f'  NEW (filter cleaned): {key}')
            else:
                new_measurements.append(idx)
                if self.debug:
                    self._log(f'  NEW: {key}')
        
        if self.debug:
            self._log(f'  Established: {len(established_measurements)}, New: {len(new_measurements)}')

        # STEP 0: LATE FUSION - Check if any established tracks from DIFFERENT radars
        # should be merged (fixes async radar detection issue)
        if self.enable_late_fusion and len(established_measurements) >= 2:
            established_measurements = self._attempt_late_fusion(all_tracks, established_measurements)

        fused_tracks = []
        processed_indices = set()

        # STEP 1: Process ESTABLISHED measurements - each goes to its own global track
        # Group by global_tid in case multiple radars see the same target
        gid_to_indices = {}
        for idx, gid in established_measurements:
            gid_to_indices.setdefault(gid, []).append(idx)
            processed_indices.add(idx)

        for gid, indices in gid_to_indices.items():
            cluster = [all_tracks[idx] for idx in indices]
            
            # CHECK FOR TRACK SPLIT: If multiple radars are updating this track
            # but their measurements are far apart, we need to split
            if len(cluster) >= 2:
                split_result = self._check_and_split_track(gid, cluster)
                if split_result is not None:
                    # Track was split - split_result contains the new fused tracks
                    fused_tracks.extend(split_result)
                    continue
            
            # Normal update (no split needed)
            if len(cluster) == 1:
                fused_tracks.append(self._ekf_update_single_with_gid(cluster[0], gid))
            else:
                fused_tracks.append(self._ekf_fuse_cluster_with_gid(cluster, gid))

        # STEP 2: For NEW measurements, check if cross-radar fusion is possible
        # BUT only if there are no existing global tracks nearby (to prevent hijacking)
        if new_measurements:
            # Get positions of all existing global tracks for proximity check
            existing_track_positions = {}
            for gid, ekf in self.filters.items():
                x = ekf.x.squeeze()
                existing_track_positions[gid] = np.array([x[0], x[1], x[2]])

            # Check if any existing tracks are close to each other (conflict zone)
            conflict_zone = self._detect_conflict_zones(existing_track_positions)
            
            if self.debug:
                self._log(f'  Existing tracks: {list(existing_track_positions.keys())}')
                self._log(f'  Conflict zones: {conflict_zone}')

            # Process new measurements
            new_clusters = self._associate_new_measurements(
                all_tracks, new_measurements, existing_track_positions, conflict_zone
            )
            
            if self.debug:
                for ci, cluster_indices in enumerate(new_clusters):
                    radars = [all_tracks[idx]['radar_name'] for idx in cluster_indices]
                    self._log(f'  New cluster {ci}: indices={cluster_indices}, radars={radars}')

            for cluster_indices in new_clusters:
                # Skip if any index in this cluster was already processed
                if any(idx in processed_indices for idx in cluster_indices):
                    continue
                
                # Mark all indices as processed
                for idx in cluster_indices:
                    processed_indices.add(idx)

                cluster = [all_tracks[idx] for idx in cluster_indices]
                
                if len(cluster) == 1:
                    fused_tracks.append(self._ekf_update_single(cluster[0]))
                else:
                    # Only fuse if from different radars
                    radars = set(t['radar_name'] for t in cluster)
                    if len(radars) >= 2:
                        fused_tracks.append(self._ekf_fuse_cluster(cluster))
                    else:
                        # Same radar - should not happen, but handle gracefully
                        for t in cluster:
                            fused_tracks.append(self._ekf_update_single(t))

        return fused_tracks

    def _check_and_split_track(self, gid, cluster):
        """
        Check if a global track should be split because measurements from different
        radars have diverged (indicating two people who were merged are now separating).
        
        Args:
            gid: The global track ID
            cluster: List of measurements from different radars assigned to this track
            
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
        needs_split = (max_pos_dist > self.split_distance_threshold or 
                       max_vel_dist > self.split_velocity_threshold)
        
        if not needs_split:
            return None
        
        # SPLIT THE TRACK
        self._log(f'SPLIT detected for GID {gid}: pos_dist={max_pos_dist:.2f}m, vel_dist={max_vel_dist:.2f}m/s')
        
        # Strategy: Keep the original GID for the radar with the oldest association,
        # create new GIDs for other radars
        fused_results = []
        
        # Find which radar had the original association (check association order)
        original_radar = None
        for radar_name in radar_names:
            key = (radar_name, radar_tracks[radar_name]['tid'])
            if key in self.radar_to_global_tid and self.radar_to_global_tid[key] == gid:
                # This radar was associated with this gid
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
                    fused_results.append(self._ekf_update_single_with_gid(tracks[0], gid))
                else:
                    fused_results.append(self._ekf_fuse_cluster_with_gid(tracks, gid))
            else:
                # Create new GID for this radar - break the association
                for t in tracks:
                    key = (t['radar_name'], t['tid'])
                    # Remove old association
                    if key in self.radar_to_global_tid:
                        del self.radar_to_global_tid[key]
                    # Create as new track
                    fused_results.append(self._ekf_update_single(t))
        
        return fused_results

    def _attempt_late_fusion(self, all_tracks, established_measurements):
        """
        LATE FUSION: Merge established tracks from DIFFERENT radars if they are close.
        
        This fixes the async detection problem:
        - Radar_A detects person first -> gets GID 1
        - Radar_B detects same person 100ms later -> gets GID 2
        - Now they're locked to separate GIDs forever
        
        Solution: If GID 1 and GID 2 are consistently close (same person), merge them.
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
            if gid in self.filters:
                x = self.filters[gid].x.squeeze()
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
                
                if pos_dist <= self.late_fusion_distance and vel_dist <= self.late_fusion_velocity:
                    merge_pairs.append((gid1, gid2, pos_dist, vel_dist))
                    if self.debug:
                        self._log(f'  LATE FUSION candidate: GID {gid1} + GID {gid2} '
                                  f'(pos={pos_dist:.3f}m, vel={vel_dist:.3f}m/s)')
        
        if not merge_pairs:
            return established_measurements
        
        # Process merges: keep the lower GID (older track), redirect higher GID
        for gid1, gid2, pos_dist, vel_dist in merge_pairs:
            keep_gid = min(gid1, gid2)
            remove_gid = max(gid1, gid2)
            
            self._log(f'LATE FUSION: Merging GID {remove_gid} into GID {keep_gid} '
                      f'(pos={pos_dist:.3f}m, vel={vel_dist:.3f}m/s)')
            
            # Update all radar->global mappings that pointed to remove_gid
            keys_to_update = [k for k, v in self.radar_to_global_tid.items() if v == remove_gid]
            for key in keys_to_update:
                self.radar_to_global_tid[key] = keep_gid
                if self.debug:
                    self._log(f'    Remapped {key} -> GID {keep_gid}')
            
            # Merge EKF state: fuse the two filters
            if remove_gid in self.filters and keep_gid in self.filters:
                # Simple approach: keep the filter with lower covariance trace
                # (more confident estimate)
                keep_trace = np.trace(self.filters[keep_gid].P)
                remove_trace = np.trace(self.filters[remove_gid].P)
                
                if remove_trace < keep_trace:
                    # The removed filter is more confident - use its state
                    self.filters[keep_gid].x = self.filters[remove_gid].x.copy()
                    self.filters[keep_gid].P = self.filters[remove_gid].P.copy()
                
                # Delete the removed filter
                del self.filters[remove_gid]
            
            # Clean up remove_gid from tracking dicts
            if remove_gid in self.global_tid_last_seen:
                del self.global_tid_last_seen[remove_gid]
            if remove_gid in self.global_tid_last_ts:
                del self.global_tid_last_ts[remove_gid]
        
        # Rebuild established_measurements with updated GIDs
        new_established = []
        for idx, old_gid in established_measurements:
            track = all_tracks[idx]
            key = (track['radar_name'], track['tid'])
            new_gid = self.radar_to_global_tid.get(key, old_gid)
            new_established.append((idx, new_gid))
        
        return new_established

    def _detect_conflict_zones(self, existing_track_positions):
        """
        Detect areas where multiple existing global tracks are close together.
        In these zones, we should NOT allow new cross-radar fusion to prevent
        accidentally merging distinct people.
        
        Returns: set of (gid1, gid2) pairs that are in conflict
        """
        conflicts = set()
        gids = list(existing_track_positions.keys())
        
        for i, gid1 in enumerate(gids):
            for gid2 in gids[i+1:]:
                pos1 = existing_track_positions[gid1]
                pos2 = existing_track_positions[gid2]
                dist = np.linalg.norm(pos1 - pos2)
                
                if dist < self.min_track_separation * 3:  # Conflict zone is 3x min separation
                    conflicts.add((min(gid1, gid2), max(gid1, gid2)))
        
        return conflicts

    def _associate_new_measurements(self, all_tracks, new_indices, existing_positions, conflict_zone):
        """
        Associate new measurements, with awareness of conflict zones.
        
        Rules:
        1. If a new measurement is near an existing track that's in a conflict zone,
           DON'T fuse it cross-radar - let each radar maintain its own track
        2. Only allow cross-radar fusion in "clear" areas
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
                if dist < self.min_track_separation * 2:
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
                continue

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
        safe_clusters = self._cross_radar_associate(all_tracks, by_radar)
        clusters.extend(safe_clusters)

        return clusters

    def _cross_radar_associate(self, all_tracks, by_radar):
        """
        Associate unassociated measurements across different radars.
        Only creates clusters from measurements belonging to DIFFERENT radars.
        Uses both position and velocity consistency.
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

                if pos_dist > self.distance_threshold:
                    if self.debug:
                        self._log(f'    Reject merge {t_i["radar_name"]}:{t_i["tid"]} <-> {t_j["radar_name"]}:{t_j["tid"]}: pos_dist={pos_dist:.3f}m > {self.distance_threshold}m')
                    continue

                # Check velocity consistency
                vel_i = np.array([t_i.get('velX', 0), t_i.get('velY', 0), t_i.get('velZ', 0)])
                vel_j = np.array([t_j.get('velX', 0), t_j.get('velY', 0), t_j.get('velZ', 0)])
                vel_dist = np.linalg.norm(vel_i - vel_j)

                if vel_dist > self.velocity_threshold:
                    if self.debug:
                        self._log(f'    Reject merge {t_i["radar_name"]}:{t_i["tid"]} <-> {t_j["radar_name"]}:{t_j["tid"]}: vel_dist={vel_dist:.3f}m/s > {self.velocity_threshold}m/s')
                    continue

                if self.debug:
                    self._log(f'    Allow merge {t_i["radar_name"]}:{t_i["tid"]} <-> {t_j["radar_name"]}:{t_j["tid"]}: pos={pos_dist:.3f}m, vel={vel_dist:.3f}m/s')
                can_merge[i, j] = True
                can_merge[j, i] = True

        # Connected components on the affinity graph
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

    # ---------- EKF-based fusion helpers ----------

    def _ekf_update_single_with_gid(self, track, gid):
        """
        Update an existing global track with a measurement.
        The global_tid is already known (locked association).
        """
        if gid not in self.filters:
            self._init_filter_from_track(track, gid)

        self._predict_to(gid, track['timestamp'])
        z, R, meas_type = self._measurement_and_R(track)
        nis = self.filters[gid].update(z, R, meas_type=meas_type)

        est = self.filters[gid].get_state()
        
        fused_track = track.copy()
        fused_track.update(est)
        fused_track['global_tid'] = gid
        fused_track['num_radars_detected'] = 1
        fused_track['source_radars'] = [track['radar_name']]
        fused_track['source_tids'] = [(track['radar_name'], track['tid'])]
        fused_track['timestamp'] = float(track['timestamp'])
        fused_track['P_trace'] = float(np.trace(self.filters[gid].P))
        fused_track['nis'] = nis
        
        self.global_tid_last_seen[gid] = time.time()
        return fused_track

    def _ekf_fuse_cluster_with_gid(self, tracks, gid):
        """
        Fuse multiple measurements into an existing global track.
        The global_tid is already known (locked association).
        """
        if len(tracks) == 1:
            return self._ekf_update_single_with_gid(tracks[0], gid)

        if gid not in self.filters:
            best = max(tracks, key=lambda t: t.get('confidence', 0.5))
            self._init_filter_from_track(best, gid)

        ts = float(np.mean([t['timestamp'] for t in tracks]))
        self._predict_to(gid, ts)

        used = []
        gated_out = []
        for t in sorted(tracks, key=lambda x: x.get('confidence', 0.5), reverse=True):
            z, R, meas_type = self._measurement_and_R(t)

            if self.use_mahalanobis_gating:
                if meas_type == 'pos_vel_acc':
                    H = np.eye(9)
                elif meas_type == 'pos_vel':
                    H = np.zeros((6, 9))
                    H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1
                    H[3, 3] = 1; H[4, 4] = 1; H[5, 5] = 1
                else:
                    H = np.zeros((3, 9))
                    H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1

                x_pred = self.filters[gid].x
                P_pred = self.filters[gid].P
                hx = H @ x_pred
                y = _as_col(z) - hx
                S = H @ P_pred @ H.T + R
                nis = float((y.T @ np.linalg.inv(S) @ y).squeeze())

                if nis > self.mahalanobis_gate:
                    gated_out.append((t['radar_name'], t['tid'], nis))
                    continue

            nis = self.filters[gid].update(z, R, meas_type=meas_type)
            used.append((t['radar_name'], t['tid'], nis))

        est = self.filters[gid].get_state()
        confidences = np.array([t.get('confidence', 0.5) for t in tracks], dtype=float)
        fused_confidence = float(np.max(confidences)) if len(confidences) else 0.5

        fused_track = {
            **est,
            'confidence': fused_confidence,
            'global_tid': gid,
            'num_radars_detected': len(tracks),
            'source_radars': [t['radar_name'] for t in tracks],
            'source_tids': [(t['radar_name'], t['tid']) for t in tracks],
            'timestamp': ts,
            'P_trace': float(np.trace(self.filters[gid].P)),
            'ekf_updates_used': used,
            'ekf_gated_out': gated_out,
        }

        self.global_tid_last_seen[gid] = time.time()
        return fused_track

    def _init_filter_from_track(self, track, global_tid):
        x0 = np.array([
            track['posX'], track['posY'], track['posZ'],
            track.get('velX', 0.0), track.get('velY', 0.0), track.get('velZ', 0.0),
            track.get('accX', 0.0), track.get('accY', 0.0), track.get('accZ', 0.0),
        ], dtype=float)

        # Initial covariance (larger uncertainty for acceleration)
        P0 = np.diag([
            self.meas_sigma_pos**2, self.meas_sigma_pos**2, self.meas_sigma_pos**2,
            (2*self.meas_sigma_vel)**2, (2*self.meas_sigma_vel)**2, (2*self.meas_sigma_vel)**2,
            (2*self.meas_sigma_acc)**2, (2*self.meas_sigma_acc)**2, (2*self.meas_sigma_acc)**2,
        ])

        self.filters[global_tid] = EKFTrack(x0=x0, P0=P0, sigma_j=self.sigma_j)
        self.global_tid_last_ts[global_tid] = float(track['timestamp'])

    def _predict_to(self, global_tid, ts):
        ts = float(ts)
        last_ts = self.global_tid_last_ts.get(global_tid, None)
        if last_ts is None:
            self.global_tid_last_ts[global_tid] = ts
            return
        dt = max(0.0, ts - float(last_ts))
        self.filters[global_tid].predict(dt)
        self.global_tid_last_ts[global_tid] = ts

    def _measurement_and_R(self, track):
        # Check what measurements are available
        has_vel = all(k in track for k in ('velX', 'velY', 'velZ'))
        has_acc = all(k in track for k in ('accX', 'accY', 'accZ'))
        
        if has_vel and has_acc:
            z = np.array([
                track['posX'], track['posY'], track['posZ'],
                track['velX'], track['velY'], track['velZ'],
                track['accX'], track['accY'], track['accZ']
            ], dtype=float)
            R = np.diag([
                self.meas_sigma_pos**2, self.meas_sigma_pos**2, self.meas_sigma_pos**2,
                self.meas_sigma_vel**2, self.meas_sigma_vel**2, self.meas_sigma_vel**2,
                self.meas_sigma_acc**2, self.meas_sigma_acc**2, self.meas_sigma_acc**2,
            ])
            return z, R, 'pos_vel_acc'
        elif has_vel:
            z = np.array([
                track['posX'], track['posY'], track['posZ'],
                track['velX'], track['velY'], track['velZ']
            ], dtype=float)
            R = np.diag([
                self.meas_sigma_pos**2, self.meas_sigma_pos**2, self.meas_sigma_pos**2,
                self.meas_sigma_vel**2, self.meas_sigma_vel**2, self.meas_sigma_vel**2,
            ])
            return z, R, 'pos_vel'
        else:
            z = np.array([track['posX'], track['posY'], track['posZ']], dtype=float)
            R = np.diag([self.meas_sigma_pos**2]*3)
            return z, R, 'pos'

    def _ekf_update_tid(self, global_tid, track):
        # Predict to measurement timestamp then update
        if global_tid not in self.filters:
            self._init_filter_from_track(track, global_tid)

        self._predict_to(global_tid, track['timestamp'])
        z, R, meas_type = self._measurement_and_R(track)

        nis = self.filters[global_tid].update(z, R, meas_type=meas_type)
        return nis

    def _ekf_update_single(self, track):
        # Keep your original global ID consistency by radar/local tid
        fused_track = self._assign_global_tid(track)
        gid = fused_track['global_tid']

        # EKF update
        self._ekf_update_tid(gid, track)

        # Output fused estimate (from EKF state)
        est = self.filters[gid].get_state()
        fused_track.update(est)

        fused_track['num_radars_detected'] = 1
        fused_track['source_radars'] = [track['radar_name']]
        fused_track['source_tids'] = [(track['radar_name'], track['tid'])]
        fused_track['timestamp'] = float(track['timestamp'])
        fused_track['P_trace'] = float(np.trace(self.filters[gid].P))
        return fused_track

    def _ekf_fuse_cluster(self, tracks):
        """
        Fuse a set of measurements (likely same target) into a single global EKF track.
        We assign/choose a global_tid, predict to timestamp, then sequentially update with each measurement.
        """
        if len(tracks) == 1:
            return self._ekf_update_single(tracks[0])

        # Choose / assign global id for the cluster (keep your old behavior)
        fused_stub = {}
        fused_stub = self._assign_global_tid_multi(tracks, fused_stub)
        gid = fused_stub['global_tid']

        # If EKF doesn’t exist yet, seed from the "best" measurement (highest confidence)
        if gid not in self.filters:
            best = max(tracks, key=lambda t: t.get('confidence', 0.5))
            self._init_filter_from_track(best, gid)

        # Predict to a representative time (use mean ts like you did, or max for causality)
        ts = float(np.mean([t['timestamp'] for t in tracks]))
        self._predict_to(gid, ts)

        # Sequential EKF updates; optionally gate outliers
        used = []
        gated_out = []
        for t in sorted(tracks, key=lambda x: x.get('confidence', 0.5), reverse=True):
            z, R, meas_type = self._measurement_and_R(t)

            # Optional Mahalanobis gating
            if self.use_mahalanobis_gating:
                # Build H based on measurement type
                if meas_type == 'pos_vel_acc':
                    H = np.eye(9)
                elif meas_type == 'pos_vel':
                    H = np.zeros((6, 9))
                    H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1
                    H[3, 3] = 1; H[4, 4] = 1; H[5, 5] = 1
                else:
                    H = np.zeros((3, 9))
                    H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1

                x_pred = self.filters[gid].x
                P_pred = self.filters[gid].P
                hx = H @ x_pred
                y = _as_col(z) - hx
                S = H @ P_pred @ H.T + R
                nis = float((y.T @ np.linalg.inv(S) @ y).squeeze())

                if nis > self.mahalanobis_gate:
                    gated_out.append((t['radar_name'], t['tid'], nis))
                    continue

            nis = self.filters[gid].update(z, R, meas_type=meas_type)
            used.append((t['radar_name'], t['tid'], nis))

        # Build fused output from EKF state
        est = self.filters[gid].get_state()
        confidences = np.array([t.get('confidence', 0.5) for t in tracks], dtype=float)
        fused_confidence = float(np.max(confidences)) if len(confidences) else 0.5

        fused_track = {
            **est,
            'confidence': fused_confidence,
            'global_tid': gid,
            'num_radars_detected': len(tracks),
            'source_radars': [t['radar_name'] for t in tracks],
            'source_tids': [(t['radar_name'], t['tid']) for t in tracks],
            'timestamp': ts,
            'P_trace': float(np.trace(self.filters[gid].P)),
            'ekf_updates_used': used,
            'ekf_gated_out': gated_out,
        }

        # Update the mapping for source tracks (so future singles keep same gid)
        for t in tracks:
            self.radar_to_global_tid[(t['radar_name'], t['tid'])] = gid
        self.global_tid_last_seen[gid] = time.time()

        return fused_track

    # ---------- ID Management ----------

    def _assign_global_tid(self, track):
        radar_name = track['radar_name']
        local_tid = track['tid']
        key = (radar_name, local_tid)

        if key in self.radar_to_global_tid:
            global_tid = self.radar_to_global_tid[key]
        else:
            global_tid = self.next_global_tid
            self.radar_to_global_tid[key] = global_tid
            self.next_global_tid += 1

        self.global_tid_last_seen[global_tid] = time.time()

        track['global_tid'] = global_tid
        track['num_radars_detected'] = 1
        track['source_radars'] = [radar_name]
        track['source_tids'] = [key]
        return track

    def _assign_global_tid_multi(self, source_tracks, fused_track):
        existing = []
        for t in source_tracks:
            key = (t['radar_name'], t['tid'])
            if key in self.radar_to_global_tid:
                existing.append(self.radar_to_global_tid[key])

        if existing:
            global_tid = min(existing)  # oldest ID
        else:
            global_tid = self.next_global_tid
            self.next_global_tid += 1

        for t in source_tracks:
            self.radar_to_global_tid[(t['radar_name'], t['tid'])] = global_tid

        self.global_tid_last_seen[global_tid] = time.time()
        fused_track['global_tid'] = global_tid
        return fused_track

    def _cleanup_old_tids(self):
        now = time.time()
        old = [tid for tid, last in self.global_tid_last_seen.items()
               if now - last > self.tid_timeout]

        for tid in old:
            del self.global_tid_last_seen[tid]

            keys_to_remove = [k for k, v in self.radar_to_global_tid.items() if v == tid]
            for k in keys_to_remove:
                del self.radar_to_global_tid[k]

            # Also drop EKF state
            if tid in self.filters:
                del self.filters[tid]
            if tid in self.global_tid_last_ts:
                del self.global_tid_last_ts[tid]

    def _log(self, txt):
        print(f'[TrackFusion]\t{txt}')


class TrackFusionVisualizer:
    def __init__(self):
        self.track_history = {}
        self.max_history = 50

    def update(self, fused_tracks):
        for track in fused_tracks:
            tid = track['global_tid']
            pos = (track['posX'], track['posY'], track['posZ'])
            self.track_history.setdefault(tid, []).append(pos)
            if len(self.track_history[tid]) > self.max_history:
                self.track_history[tid].pop(0)

    def get_track_trails(self):
        return self.track_history


if __name__ == '__main__':
    fusion_cfg = {
        'TRACK_FUSION_CFG': {
            # Association thresholds (for NEW tracks to merge cross-radar)
            'distance_threshold': 0.5,      # meters - max position distance
            'velocity_threshold': 1.0,      # m/s - max velocity difference
            
            # Track identity preservation (prevents merging when close)
            'min_track_separation': 0.3,    # meters - conflict zone threshold
            
            # Track splitting (recovers from incorrect merges)
            'split_distance_threshold': 0.8,  # meters - split when diverging
            'split_velocity_threshold': 0.8,  # m/s - split on velocity divergence
            
            # EKF process noise (Constant Acceleration model)
            'sigma_j': 1.0,                 # m/s^3 - jerk std dev
            
            # Measurement noise
            'meas_sigma_pos': 0.25,         # meters
            'meas_sigma_vel': 0.50,         # m/s
            'meas_sigma_acc': 1.0,          # m/s^2
            
            # Outlier rejection
            'use_mahalanobis_gating': True,
            'mahalanobis_gate': 22.0,       # chi2(9)@99% for full state gating
            
            # Track management
            'tid_timeout': 2.0,             # seconds
        },
        'RADAR_CFG_LIST': [{'name': 'Radar1'}, {'name': 'Radar2'}, {'name': 'Radar3'}]
    }

    fusion = TrackFusion(**fusion_cfg)

    t0 = time.time()
    frame1 = {
        'radar_name': 'Radar1',
        'timestamp': t0,
        'tracks': [
            {'tid': 1, 'posX': 1.0, 'posY': 2.0, 'posZ': 1.5,
             'velX': 0.1, 'velY': 0.2, 'velZ': 0.0,
             'confidence': 0.9},
            {'tid': 2, 'posX': 3.0, 'posY': 4.0, 'posZ': 1.2,
             'velX': -0.1, 'velY': 0.0, 'velZ': 0.0,
             'confidence': 0.7}
        ]
    }

    frame2 = {
        'radar_name': 'Radar2',
        'timestamp': t0 + 0.02,
        'tracks': [
            {'tid': 5, 'posX': 1.1, 'posY': 2.1, 'posZ': 1.4,
             'velX': 0.15, 'velY': 0.18, 'velZ': 0.0,
             'confidence': 0.85}
        ]
    }

    frame3 = {
        'radar_name': 'Radar3',
        'timestamp': t0 + 0.04,
        'tracks': [
            {'tid': 10, 'posX': 1.05, 'posY': 2.05, 'posZ': 1.45,
             'velX': 0.12, 'velY': 0.19, 'velZ': 0.0,
             'confidence': 0.88},
            {'tid': 11, 'posX': 5.0, 'posY': 6.0, 'posZ': 1.0,
             'velX': 0.5, 'velY': -0.3, 'velZ': 0.0,
             'confidence': 0.75}
        ]
    }

    fused = fusion.fuse_tracks([frame1, frame2, frame3])
    print(f'\nFused {len(fused)} tracks:')
    for tr in fused:
        print(f"  GID {tr['global_tid']}: "
              f"Pos=({tr['posX']:.2f}, {tr['posY']:.2f}, {tr['posZ']:.2f}) "
              f"Vel=({tr['velX']:.2f}, {tr['velY']:.2f}, {tr['velZ']:.2f}) "
              f"Conf={tr.get('confidence', 0.5):.2f} "
              f"P_tr={tr.get('P_trace', np.nan):.3f} "
              f"radars={tr.get('num_radars_detected', 1)} "
              f"sources={tr.get('source_radars', [])}")

