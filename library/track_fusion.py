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

7. tid_timeout (default: 2.0) - Track ID timeout [seconds]
   
   How long to keep a track alive after last detection before deleting it.
   
   - INCREASE for intermittent detections or slow-moving targets
   - DECREASE for fast update rates or to quickly drop lost tracks

EXAMPLE CONFIGURATION:
----------------------
fusion_cfg = {
    'TRACK_FUSION_CFG': {
        'distance_threshold': 0.5,   # meters
        'sigma_j': 1.0,              # m/s^3 - increase for maneuvering targets
        'meas_sigma_pos': 0.25,      # m - match to radar spec
        'meas_sigma_vel': 0.50,      # m/s - match to radar spec
        'meas_sigma_acc': 1.0,       # m/s^2 - usually higher uncertainty
        'use_mahalanobis_gating': True,
        'mahalanobis_gate': 22.0,    # chi2(9) @ 99%
        'tid_timeout': 2.0,          # seconds
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
from scipy.spatial.distance import cdist
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
        self.last_ts = None

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

        # Global EKF bank
        self.filters = {}                  # global_tid -> EKFTrack
        self.global_tid_last_ts = {}       # global_tid -> last measurement timestamp

        self._log('Track Fusion (EKF-CA) initialized')

    def fuse_tracks(self, radar_frames):
        if not radar_frames:
            return []

        self._cleanup_old_tids()

        # Flatten measurements
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
        n = len(all_tracks)

        # If only one track, fast path
        if n == 1:
            return [self._ekf_update_single(all_tracks[0])]

        # Build distance matrix on positions (for quick clustering)
        positions = np.array([[t['posX'], t['posY'], t['posZ']] for t in all_tracks], dtype=float)
        dist_matrix = cdist(positions, positions, metric='euclidean')

        fused_tracks = []
        visited = set()

        # Connected components: any tracks transitively within threshold get merged
        for i in range(n):
            if i in visited:
                continue

            stack = [i]
            cluster_indices = []

            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                cluster_indices.append(cur)

                neighbors = np.where(dist_matrix[cur, :] < self.distance_threshold)[0]
                for nb in neighbors:
                    if nb != cur and nb not in visited:
                        stack.append(nb)

            cluster = [all_tracks[idx] for idx in cluster_indices]

            if len(cluster) > 1:
                fused_tracks.append(self._ekf_fuse_cluster(cluster))
            else:
                fused_tracks.append(self._ekf_update_single(cluster[0]))

        return fused_tracks

    # ---------- EKF-based fusion helpers ----------

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
            'distance_threshold': 0.5,
            'sigma_a': 2.0,
            'meas_sigma_pos': 0.25,
            'meas_sigma_vel': 0.50,
            'use_mahalanobis_gating': True,
            'mahalanobis_gate': 16.0,
            'tid_timeout': 2.0
        },
        'RADAR_CFG_LIST': [{'name': 'Radar1'}, {'name': 'Radar2'}]
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

    fused = fusion.fuse_tracks([frame1, frame2])
    print(f'\nFused {len(fused)} tracks:')
    for tr in fused:
        print(f"  GID {tr['global_tid']}: "
              f"Pos=({tr['posX']:.2f}, {tr['posY']:.2f}, {tr['posZ']:.2f}) "
              f"Vel=({tr['velX']:.2f}, {tr['velY']:.2f}, {tr['velZ']:.2f}) "
              f"Conf={tr.get('confidence', 0.5):.2f} "
              f"P_tr={tr.get('P_trace', np.nan):.3f} "
              f"radars={tr.get('num_radars_detected', 1)}")

