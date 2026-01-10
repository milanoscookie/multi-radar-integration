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
   - INCREASE (2.0-5.0) for highly maneuvering targets
   - DECREASE (0.1-0.5) for smooth trajectories

2. meas_sigma_pos (default: 0.25) - Position measurement noise [m]
   - Set to your radar's position accuracy specification

3. meas_sigma_vel (default: 0.50) - Velocity measurement noise [m/s]
   - Typically 2-3x the position noise

4. meas_sigma_acc (default: 1.0) - Acceleration measurement noise [m/s^2]
   - Usually higher uncertainty for radar-derived acceleration

5. mahalanobis_gate (default: 22.0) - Gating threshold for outlier rejection
   - chi2(9) @ 99% ≈ 21.7

6. distance_threshold (default: 0.5) - Track association distance [m]

7. velocity_threshold (default: 1.0) - Velocity consistency threshold [m/s]

8. tid_timeout (default: 2.0) - Track ID timeout [seconds]

9. min_track_separation (default: 0.3) - Conflict zone threshold [m]

10. split_distance_threshold (default: 0.8) - Track split position threshold [m]

11. split_velocity_threshold (default: 0.8) - Track split velocity threshold [m/s]

12. enable_late_fusion (default: True) - Enable late fusion for async detection

13. late_fusion_distance (default: 0.5) - Late fusion position threshold [m]

14. late_fusion_velocity (default: 1.0) - Late fusion velocity threshold [m/s]

================================================================================
"""

import numpy as np
import time

from .ekf import EKFTrack, _as_col, build_measurement_matrix
from .id_manager import TrackIDManager
from .association import (
    detect_conflict_zones,
    associate_new_measurements,
    attempt_late_fusion,
)
from .splitting import check_and_split_track


class TrackFusion:
    """Main track fusion class for multi-radar systems."""
    
    def __init__(self, **kwargs_CFG):
        self.fusion_cfg = kwargs_CFG.get('TRACK_FUSION_CFG', {})
        self.radar_cfg_list = kwargs_CFG['RADAR_CFG_LIST']

        # Association thresholds
        self.distance_threshold = self.fusion_cfg.get('distance_threshold', 0.5)  # meters
        self.velocity_threshold = self.fusion_cfg.get('velocity_threshold', 1.0)  # m/s

        # EKF knobs (CA model)
        self.sigma_j = self.fusion_cfg.get('sigma_j', 1.0)    # m/s^3 process jerk std
        self.meas_sigma_pos = self.fusion_cfg.get('meas_sigma_pos', 0.25)  # m
        self.meas_sigma_vel = self.fusion_cfg.get('meas_sigma_vel', 0.50)  # m/s
        self.meas_sigma_acc = self.fusion_cfg.get('meas_sigma_acc', 1.0)   # m/s^2
        self.use_mahalanobis_gating = self.fusion_cfg.get('use_mahalanobis_gating', True)
        self.mahalanobis_gate = self.fusion_cfg.get('mahalanobis_gate', 22.0)

        # Track ID management
        tid_timeout = float(self.fusion_cfg.get('tid_timeout', 2.0))
        self.id_manager = TrackIDManager(tid_timeout=tid_timeout)

        # Conflict zone threshold
        self.min_track_separation = float(self.fusion_cfg.get('min_track_separation', 0.3))

        # Track splitting parameters
        self.split_distance_threshold = float(self.fusion_cfg.get('split_distance_threshold', 0.8))
        self.split_velocity_threshold = float(self.fusion_cfg.get('split_velocity_threshold', 0.8))

        # Late fusion parameters
        self.enable_late_fusion = self.fusion_cfg.get('enable_late_fusion', True)
        self.late_fusion_distance = float(self.fusion_cfg.get('late_fusion_distance', 0.5))
        self.late_fusion_velocity = float(self.fusion_cfg.get('late_fusion_velocity', 1.0))

        # Global EKF bank
        self.filters = {}                  # global_tid -> EKFTrack
        self.global_tid_last_ts = {}       # global_tid -> last measurement timestamp

        # Debug mode
        self.debug = self.fusion_cfg.get('debug', False)

        self._log('Track Fusion (EKF-CA) initialized with track lock-in and split detection')

    def fuse_tracks(self, radar_frames):
        """
        Main fusion method with TRACK IDENTITY PRESERVATION.
        
        Args:
            radar_frames: list of frame dicts, each with:
                - radar_name: str
                - timestamp: float
                - tracks: list of track dicts
                
        Returns:
            list of fused track dicts
        """
        if not radar_frames:
            return []

        self.id_manager.cleanup_old_tids(self.filters, self.global_tid_last_ts)

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

        # Separate measurements into ESTABLISHED and NEW
        established_measurements = []  # [(idx, gid), ...]
        new_measurements = []          # [idx, ...]

        for idx, track in enumerate(all_tracks):
            gid = self.id_manager.get_global_tid(track['radar_name'], track['tid'])
            if gid is not None and gid in self.filters:
                established_measurements.append((idx, gid))
                if self.debug:
                    self._log(f'  ESTABLISHED: ({track["radar_name"]}, {track["tid"]}) -> GID {gid}')
            else:
                new_measurements.append(idx)
                if self.debug:
                    self._log(f'  NEW: ({track["radar_name"]}, {track["tid"]})')
        
        if self.debug:
            self._log(f'  Established: {len(established_measurements)}, New: {len(new_measurements)}')

        # STEP 0: LATE FUSION - merge established tracks from different radars if close
        if self.enable_late_fusion and len(established_measurements) >= 2:
            established_measurements = attempt_late_fusion(
                all_tracks, established_measurements, self.filters, self.id_manager,
                self.global_tid_last_ts, self.late_fusion_distance, self.late_fusion_velocity,
                self.debug, self._log
            )

        fused_tracks = []
        processed_indices = set()

        # STEP 1: Process ESTABLISHED measurements
        gid_to_indices = {}
        for idx, gid in established_measurements:
            gid_to_indices.setdefault(gid, []).append(idx)
            processed_indices.add(idx)

        for gid, indices in gid_to_indices.items():
            cluster = [all_tracks[idx] for idx in indices]
            
            # Check for track split
            if len(cluster) >= 2:
                split_result = check_and_split_track(
                    gid, cluster, self.id_manager, self.filters, self.global_tid_last_ts,
                    self.split_distance_threshold, self.split_velocity_threshold,
                    self._ekf_update_single, self._ekf_update_single_with_gid,
                    self._ekf_fuse_cluster_with_gid, self._log
                )
                if split_result is not None:
                    fused_tracks.extend(split_result)
                    continue
            
            # Normal update
            if len(cluster) == 1:
                fused_tracks.append(self._ekf_update_single_with_gid(cluster[0], gid))
            else:
                fused_tracks.append(self._ekf_fuse_cluster_with_gid(cluster, gid))

        # STEP 2: Process NEW measurements
        if new_measurements:
            existing_track_positions = {}
            for gid, ekf in self.filters.items():
                x = ekf.x.squeeze()
                existing_track_positions[gid] = np.array([x[0], x[1], x[2]])

            conflict_zone = detect_conflict_zones(existing_track_positions, self.min_track_separation)
            
            if self.debug:
                self._log(f'  Existing tracks: {list(existing_track_positions.keys())}')
                self._log(f'  Conflict zones: {conflict_zone}')

            new_clusters = associate_new_measurements(
                all_tracks, new_measurements, existing_track_positions, conflict_zone,
                self.min_track_separation, self.distance_threshold, self.velocity_threshold,
                self.debug, self._log
            )
            
            if self.debug:
                for ci, cluster_indices in enumerate(new_clusters):
                    radars = [all_tracks[idx]['radar_name'] for idx in cluster_indices]
                    self._log(f'  New cluster {ci}: indices={cluster_indices}, radars={radars}')

            for cluster_indices in new_clusters:
                if any(idx in processed_indices for idx in cluster_indices):
                    continue
                
                for idx in cluster_indices:
                    processed_indices.add(idx)

                cluster = [all_tracks[idx] for idx in cluster_indices]
                
                if len(cluster) == 1:
                    fused_tracks.append(self._ekf_update_single(cluster[0]))
                else:
                    radars = set(t['radar_name'] for t in cluster)
                    if len(radars) >= 2:
                        fused_tracks.append(self._ekf_fuse_cluster(cluster))
                    else:
                        for t in cluster:
                            fused_tracks.append(self._ekf_update_single(t))

        return fused_tracks

    # ---------- EKF-based fusion helpers ----------

    def _ekf_update_single_with_gid(self, track, gid):
        """Update an existing global track with a measurement."""
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
        
        self.id_manager.update_last_seen(gid)
        return fused_track

    def _ekf_fuse_cluster_with_gid(self, tracks, gid):
        """Fuse multiple measurements into an existing global track."""
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
                H = build_measurement_matrix(meas_type)
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

        self.id_manager.update_last_seen(gid)
        return fused_track

    def _ekf_update_single(self, track):
        """Create/update a track for a single measurement."""
        fused_track = self.id_manager.assign_global_tid(track)
        gid = fused_track['global_tid']

        if gid not in self.filters:
            self._init_filter_from_track(track, gid)

        self._predict_to(gid, track['timestamp'])
        z, R, meas_type = self._measurement_and_R(track)
        self.filters[gid].update(z, R, meas_type=meas_type)

        est = self.filters[gid].get_state()
        fused_track.update(est)

        fused_track['num_radars_detected'] = 1
        fused_track['source_radars'] = [track['radar_name']]
        fused_track['source_tids'] = [(track['radar_name'], track['tid'])]
        fused_track['timestamp'] = float(track['timestamp'])
        fused_track['P_trace'] = float(np.trace(self.filters[gid].P))
        return fused_track

    def _ekf_fuse_cluster(self, tracks):
        """Fuse a cluster of measurements into a new global track."""
        if len(tracks) == 1:
            return self._ekf_update_single(tracks[0])

        fused_stub = {}
        fused_stub = self.id_manager.assign_global_tid_multi(tracks, fused_stub)
        gid = fused_stub['global_tid']

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
                H = build_measurement_matrix(meas_type)
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

        self.id_manager.update_last_seen(gid)
        return fused_track

    def _init_filter_from_track(self, track, global_tid):
        """Initialize an EKF from a track measurement."""
        x0 = np.array([
            track['posX'], track['posY'], track['posZ'],
            track.get('velX', 0.0), track.get('velY', 0.0), track.get('velZ', 0.0),
            track.get('accX', 0.0), track.get('accY', 0.0), track.get('accZ', 0.0),
        ], dtype=float)

        P0 = np.diag([
            self.meas_sigma_pos**2, self.meas_sigma_pos**2, self.meas_sigma_pos**2,
            (2*self.meas_sigma_vel)**2, (2*self.meas_sigma_vel)**2, (2*self.meas_sigma_vel)**2,
            (2*self.meas_sigma_acc)**2, (2*self.meas_sigma_acc)**2, (2*self.meas_sigma_acc)**2,
        ])

        self.filters[global_tid] = EKFTrack(x0=x0, P0=P0, sigma_j=self.sigma_j)
        self.global_tid_last_ts[global_tid] = float(track['timestamp'])

    def _predict_to(self, global_tid, ts):
        """Predict filter state to a given timestamp."""
        ts = float(ts)
        last_ts = self.global_tid_last_ts.get(global_tid, None)
        if last_ts is None:
            self.global_tid_last_ts[global_tid] = ts
            return
        dt = max(0.0, ts - float(last_ts))
        self.filters[global_tid].predict(dt)
        self.global_tid_last_ts[global_tid] = ts

    def _measurement_and_R(self, track):
        """Extract measurement vector and noise covariance from track."""
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

    def _log(self, txt):
        """Log a message."""
        print(f'[TrackFusion]\t{txt}')
