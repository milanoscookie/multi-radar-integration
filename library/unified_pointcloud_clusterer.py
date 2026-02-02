"""
Unified Point Cloud Clustering Process using DBSCAN + Kalman Filter
Combines point clouds from ALL radars, then clusters to form tracks

Pipeline: [Radar1] ─┬─> [Frame Sync] -> [Unified Clusterer] -> [Kalman Filter] -> [Tracked Objects]
          [Radar2] ─┘   

Features:
- Frame synchronization across multiple radars
- DBSCAN clustering on merged point cloud
- Kalman Filter for track state estimation (position, velocity, acceleration)
"""

import time
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


class KalmanFilterCA:
    """
    Kalman Filter with Constant Acceleration (CA) motion model
    State: [x, y, z, vx, vy, vz, ax, ay, az]
    """
    
    def __init__(self, initial_pos, initial_vel=None, sigma_j=1.0, 
                 meas_sigma_pos=0.25, meas_sigma_vel=0.5, meas_sigma_acc=1.0):
        """
        Initialize Kalman Filter
        
        Args:
            initial_pos: [x, y, z] initial position
            initial_vel: [vx, vy, vz] initial velocity (optional)
            sigma_j: jerk standard deviation (process noise)
            meas_sigma_pos: position measurement noise
            meas_sigma_vel: velocity measurement noise
            meas_sigma_acc: acceleration measurement noise
        """
        # State vector: [x, y, z, vx, vy, vz, ax, ay, az]
        self.state = np.zeros(9)
        self.state[0:3] = initial_pos
        if initial_vel is not None:
            self.state[3:6] = initial_vel
        
        # State covariance - start with higher uncertainty
        self.P = np.eye(9)
        self.P[0:3, 0:3] *= meas_sigma_pos ** 2  # Position uncertainty
        self.P[3:6, 3:6] *= meas_sigma_vel ** 2  # Velocity uncertainty
        self.P[6:9, 6:9] *= meas_sigma_acc ** 2  # Acceleration uncertainty
        
        # Process noise (jerk)
        self.sigma_j = sigma_j
        
        # Measurement noise
        self.meas_sigma_pos = meas_sigma_pos
        self.meas_sigma_vel = meas_sigma_vel
        self.meas_sigma_acc = meas_sigma_acc
        
        # Measurement matrix (observe position and velocity)
        # H maps state [x,y,z,vx,vy,vz,ax,ay,az] to measurement [x,y,z,vx,vy,vz]
        self.H = np.zeros((6, 9))
        self.H[0:3, 0:3] = np.eye(3)  # Position
        self.H[3:6, 3:6] = np.eye(3)  # Velocity
        
        # Measurement noise covariance
        self.R = np.diag([
            meas_sigma_pos**2, meas_sigma_pos**2, meas_sigma_pos**2,
            meas_sigma_vel**2, meas_sigma_vel**2, meas_sigma_vel**2
        ])
        
        self.last_update_time = time.time()
    
    def _get_F(self, dt):
        """State transition matrix for CA model"""
        F = np.eye(9)
        # Position depends on velocity and acceleration
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt**2
        # Velocity depends on acceleration
        F[3:6, 6:9] = np.eye(3) * dt
        return F
    
    def _get_Q(self, dt):
        """Process noise covariance matrix (discrete white noise jerk model)"""
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        
        # Jerk-based process noise
        q = self.sigma_j ** 2
        
        Q = np.zeros((9, 9))
        
        # Block diagonal structure for x, y, z
        for i in range(3):
            Q[i, i] = dt5 / 20 * q          # pos-pos
            Q[i, i+3] = dt4 / 8 * q          # pos-vel
            Q[i, i+6] = dt3 / 6 * q          # pos-acc
            Q[i+3, i] = dt4 / 8 * q          # vel-pos
            Q[i+3, i+3] = dt3 / 3 * q        # vel-vel
            Q[i+3, i+6] = dt2 / 2 * q        # vel-acc
            Q[i+6, i] = dt3 / 6 * q          # acc-pos
            Q[i+6, i+3] = dt2 / 2 * q        # acc-vel
            Q[i+6, i+6] = dt * q             # acc-acc
        
        return Q
    
    def predict(self, dt=None):
        """Predict state forward in time"""
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_update_time
        
        if dt <= 0:
            dt = 0.05  # Default 50ms
        
        F = self._get_F(dt)
        Q = self._get_Q(dt)
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.P = F @ self.P @ F.T + Q
        
        self.last_update_time = current_time
        return self.state.copy()
    
    def update(self, measurement_pos, measurement_vel=None):
        """
        Update state with measurement
        
        Args:
            measurement_pos: [x, y, z] measured position
            measurement_vel: [vx, vy, vz] measured velocity (optional)
        """
        if measurement_vel is None:
            # Position-only update
            H = self.H[0:3, :]
            R = self.R[0:3, 0:3]
            z = np.array(measurement_pos)
        else:
            # Full position + velocity update
            H = self.H
            R = self.R
            z = np.concatenate([measurement_pos, measurement_vel])
        
        # Innovation
        y = z - H @ self.state
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        self.last_update_time = time.time()
        return self.state.copy()
    
    def get_state(self):
        """Get current state estimate"""
        return {
            'position': self.state[0:3].copy(),
            'velocity': self.state[3:6].copy(),
            'acceleration': self.state[6:9].copy()
        }
    
    def get_position(self):
        return self.state[0:3].copy()
    
    def get_velocity(self):
        return self.state[3:6].copy()
    
    def get_acceleration(self):
        return self.state[6:9].copy()


class FrameSynchronizer:
    """
    Synchronizes frames from multiple radars based on timestamps
    
    Strategies:
    - LATEST: Use latest available frame from each radar
    - SYNC_WINDOW: Collect frames within a time window
    - ALL_READY: Wait until all radars have data
    """
    
    def __init__(self, num_radars, sync_window=0.05, strategy='SYNC_WINDOW'):
        """
        Args:
            num_radars: Number of radar inputs
            sync_window: Time window in seconds for frame collection
            strategy: 'LATEST', 'SYNC_WINDOW', or 'ALL_READY'
        """
        self.num_radars = num_radars
        self.sync_window = sync_window
        self.strategy = strategy
        
        # Frame buffers - stores latest frame from each radar
        self.frame_buffers = [None] * num_radars
        self.frame_timestamps = [0.0] * num_radars
        
        # Sync statistics
        self.total_syncs = 0
        self.dropped_frames = 0
        self.radars_present = [0] * num_radars
    
    def add_frame(self, radar_idx, frame, timestamp=None):
        """Add a frame from a radar to the buffer"""
        if timestamp is None:
            timestamp = time.time()
        
        self.frame_buffers[radar_idx] = frame
        self.frame_timestamps[radar_idx] = timestamp
    
    def get_synchronized_frames(self):
        """
        Get synchronized frames based on strategy
        
        Returns:
            list of frames (or None for missing radars), reference_timestamp
        """
        current_time = time.time()
        
        if self.strategy == 'LATEST':
            # Return whatever frames we have
            return self._get_latest_frames(current_time)
        
        elif self.strategy == 'SYNC_WINDOW':
            # Return frames within sync window
            return self._get_sync_window_frames(current_time)
        
        elif self.strategy == 'ALL_READY':
            # Only return if all radars have recent data
            return self._get_all_ready_frames(current_time)
        
        return None, current_time
    
    def _get_latest_frames(self, current_time):
        """Return latest frames from each radar"""
        result = []
        for i in range(self.num_radars):
            frame = self.frame_buffers[i]
            if frame is not None:
                # Clear after reading
                self.frame_buffers[i] = None
                result.append(frame)
                self.radars_present[i] += 1
            else:
                result.append(None)
        
        if any(f is not None for f in result):
            self.total_syncs += 1
            return result, current_time
        return None, current_time
    
    def _get_sync_window_frames(self, current_time):
        """Return frames within sync window"""
        result = []
        reference_time = max(self.frame_timestamps)
        
        for i in range(self.num_radars):
            frame = self.frame_buffers[i]
            ts = self.frame_timestamps[i]
            
            if frame is not None and (reference_time - ts) <= self.sync_window:
                result.append(frame)
                self.frame_buffers[i] = None
                self.radars_present[i] += 1
            else:
                result.append(None)
                if frame is not None and (reference_time - ts) > self.sync_window:
                    # Frame too old, drop it
                    self.frame_buffers[i] = None
                    self.dropped_frames += 1
        
        if any(f is not None for f in result):
            self.total_syncs += 1
            return result, reference_time
        return None, current_time
    
    def _get_all_ready_frames(self, current_time):
        """Return frames only if all radars have data"""
        all_ready = all(f is not None for f in self.frame_buffers)
        
        if not all_ready:
            return None, current_time
        
        result = self.frame_buffers.copy()
        self.frame_buffers = [None] * self.num_radars
        
        for i in range(self.num_radars):
            self.radars_present[i] += 1
        
        self.total_syncs += 1
        return result, max(self.frame_timestamps)
    
    def get_stats(self):
        """Get synchronization statistics"""
        return {
            'total_syncs': self.total_syncs,
            'dropped_frames': self.dropped_frames,
            'radar_contributions': self.radars_present.copy()
        }


class UnifiedPointCloudClusterer:
    """
    Combines point clouds from multiple radars and clusters them together.
    Features:
    - Frame synchronization across radars
    - DBSCAN clustering on merged point cloud
    - Kalman Filter for track state estimation
    """
    
    def __init__(self, run_flag, input_queue_list, output_queue, shared_param_dict, **kwargs_CFG):
        """
        Initialize unified clusterer
        
        Args:
            run_flag: multiprocessing.Value for run control
            input_queue_list: List of queues, one per radar (point cloud frames)
            output_queue: Single output queue for tracked objects
            shared_param_dict: Shared status dictionary
            kwargs_CFG: Configuration dict with POINTCLOUD_CLUSTERING_CFG
        """
        self.run_flag = run_flag
        self.input_queue_list = input_queue_list
        self.output_queue = output_queue
        self.status = shared_param_dict['proc_status_dict']
        
        self.status['UnifiedClusterer'] = True
        self.num_radars = len(input_queue_list)
        
        # Load clustering config
        cfg = kwargs_CFG.get('POINTCLOUD_CLUSTERING_CFG', {})
        
        # DBSCAN parameters
        self.eps = cfg.get('eps', 0.5)
        self.min_samples = cfg.get('min_samples', 3)
        self.min_points_per_cluster = cfg.get('min_points_per_cluster', 3)
        
        # Physical filtering
        self.min_cluster_height = cfg.get('min_cluster_height', 0.3)
        self.max_cluster_height = cfg.get('max_cluster_height', 2.5)
        self.min_cluster_volume = cfg.get('min_cluster_volume', 0.01)
        self.max_cluster_volume = cfg.get('max_cluster_volume', 5.0)
        
        # Temporal tracking
        self.max_track_distance = cfg.get('max_track_distance', 1.5)
        self.track_timeout = cfg.get('track_timeout', 1.0)
        
        # Frame sync settings
        self.frame_collection_timeout = cfg.get('frame_collection_timeout', 0.05)  # 50ms
        self.sync_strategy = cfg.get('sync_strategy', 'SYNC_WINDOW')  # LATEST, SYNC_WINDOW, ALL_READY
        
        # Kalman Filter parameters (from TRACK_FUSION_CFG if available, else defaults)
        kf_cfg = kwargs_CFG.get('TRACK_FUSION_CFG', {})
        self.sigma_j = kf_cfg.get('sigma_j', 1.0)
        self.meas_sigma_pos = kf_cfg.get('meas_sigma_pos', 0.25)
        self.meas_sigma_vel = kf_cfg.get('meas_sigma_vel', 0.5)
        self.meas_sigma_acc = kf_cfg.get('meas_sigma_acc', 1.0)
        
        # Initialize frame synchronizer
        self.frame_sync = FrameSynchronizer(
            num_radars=self.num_radars,
            sync_window=self.frame_collection_timeout,
            strategy=self.sync_strategy
        )
        
        # Track state - now with Kalman Filters
        self.active_tracks = {}  # tid -> track_info (includes KF)
        self.next_track_id = 1
        
        # Statistics
        self.frame_count = 0
        self.total_clusters = 0
        self.total_points = 0
        
        self._log(f'Initialized UnifiedClusterer for {self.num_radars} radars')
        self._log(f'  DBSCAN: eps={self.eps}m, min_samples={self.min_samples}')
        self._log(f'  Frame sync: strategy={self.sync_strategy}, window={self.frame_collection_timeout*1000:.0f}ms')
        self._log(f'  Kalman Filter: sigma_j={self.sigma_j}, pos_noise={self.meas_sigma_pos}m')
    
    def run(self):
        """Main process loop - collect from all radars, sync, cluster, track with KF"""
        self._log('Starting unified clustering loop...')
        
        last_process_time = time.time()
        
        while self.run_flag.value:
            try:
                # Collect frames from all radar queues into synchronizer
                for i, queue in enumerate(self.input_queue_list):
                    if not queue.empty():
                        frame = queue.get(block=False)
                        timestamp = frame.get('timestamp', time.time())
                        self.frame_sync.add_frame(i, frame, timestamp)
                
                current_time = time.time()
                time_elapsed = current_time - last_process_time
                
                # Try to get synchronized frames
                if time_elapsed >= self.frame_collection_timeout:
                    synced_frames, ref_timestamp = self.frame_sync.get_synchronized_frames()
                    
                    if synced_frames is not None:
                        # Merge point clouds from synchronized frames
                        merged_points, radar_sources = self._merge_point_clouds(synced_frames)
                        
                        if len(merged_points) >= self.min_samples:
                            # Process the merged point cloud
                            tracked_objects = self._process_merged_cloud(
                                merged_points, radar_sources, ref_timestamp
                            )
                            
                            # Output tracked objects
                            if len(tracked_objects) > 0:
                                # Count contributing radars
                                radars_in_frame = sum(1 for f in synced_frames if f is not None)
                                
                                output_frame = {
                                    'frame_number': self.frame_count,
                                    'num_detected_obj': len(tracked_objects),
                                    'tracks': tracked_objects,
                                    'num_radars': self.num_radars,
                                    'radars_contributing': radars_in_frame,
                                    'total_points': len(merged_points),
                                    'timestamp': ref_timestamp,
                                    'sync_stats': self.frame_sync.get_stats()
                                }
                                self.output_queue.put(output_frame)
                            
                            self.frame_count += 1
                            self.total_points += len(merged_points)
                            
                            if self.frame_count % 100 == 0:
                                avg_points = self.total_points / max(1, self.frame_count)
                                avg_clusters = self.total_clusters / max(1, self.frame_count)
                                sync_stats = self.frame_sync.get_stats()
                                self._log(f'Frames: {self.frame_count}, '
                                         f'Avg pts: {avg_points:.1f}, '
                                         f'Avg tracks: {avg_clusters:.1f}, '
                                         f'Active: {len(self.active_tracks)}, '
                                         f'Dropped: {sync_stats["dropped_frames"]}')
                        else:
                            # Still run KF prediction for active tracks
                            self._predict_all_tracks(current_time)
                    
                    last_process_time = current_time
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                    
            except Exception as e:
                self._log(f'Error: {e}')
                import traceback
                traceback.print_exc()
                time.sleep(0.01)
        
        self._log('Stopped')
        self.status['UnifiedClusterer'] = False
    
    def _merge_point_clouds(self, radar_buffers):
        """
        Merge point clouds from all radar buffers into single array
        
        Args:
            radar_buffers: List of frame dicts (or None) from each radar
        
        Returns:
            merged_points: List of point dicts with radar_id added
            radar_sources: List of radar indices for each point
        """
        merged_points = []
        radar_sources = []
        
        for radar_idx, frame in enumerate(radar_buffers):
            if frame is None:
                continue
            
            point_cloud = frame.get('point_cloud', {})
            points = point_cloud.get('points', [])
            radar_name = frame.get('radar_name', f'Radar_{radar_idx}')
            
            for pt in points:
                # Add radar source info to each point
                merged_pt = pt.copy()
                merged_pt['radar_idx'] = radar_idx
                merged_pt['radar_name'] = radar_name
                merged_points.append(merged_pt)
                radar_sources.append(radar_idx)
        
        return merged_points, radar_sources
    
    def _process_merged_cloud(self, points, radar_sources, current_time):
        """
        Process merged point cloud: cluster and track
        
        Args:
            points: List of point dicts (merged from all radars)
            radar_sources: List of radar indices for each point
            current_time: Current timestamp
        
        Returns:
            List of tracked object dicts
        """
        if len(points) < self.min_samples:
            self._cleanup_old_tracks(current_time)
            return []
        
        # Convert to numpy arrays
        positions = np.array([[pt['x'], pt['y'], pt['z']] for pt in points])
        dopplers = np.array([pt['doppler'] for pt in points])
        snrs = np.array([pt.get('snr') or 0 for pt in points])
        radar_indices = np.array(radar_sources)
        
        # Run DBSCAN on merged point cloud
        cluster_labels = self._cluster_dbscan(positions)
        
        # Extract cluster features
        clusters = self._extract_clusters(
            cluster_labels, positions, dopplers, snrs, radar_indices, current_time
        )
        
        # Filter invalid clusters
        clusters = self._filter_clusters(clusters)
        
        # Associate with existing tracks
        tracked_objects = self._associate_tracks(clusters, current_time)
        
        # Cleanup stale tracks
        self._cleanup_old_tracks(current_time)
        
        self.total_clusters += len(tracked_objects)
        
        return tracked_objects
    
    def _cluster_dbscan(self, positions):
        """Apply DBSCAN clustering to positions"""
        if len(positions) < self.min_samples:
            return np.full(len(positions), -1)
        
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=1)
        return dbscan.fit_predict(positions)
    
    def _extract_clusters(self, labels, positions, dopplers, snrs, radar_indices, timestamp):
        """
        Extract features for each cluster
        
        Returns list of cluster dicts with:
        - centroid, velocity, dimensions, volume
        - num_points, radar_coverage (which radars contributed)
        """
        unique_labels = set(labels)
        clusters = []
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            mask = labels == label
            cluster_pos = positions[mask]
            cluster_doppler = dopplers[mask]
            cluster_snr = snrs[mask]
            cluster_radars = radar_indices[mask]
            
            num_points = len(cluster_pos)
            if num_points < self.min_points_per_cluster:
                continue
            
            # Centroid
            centroid = np.mean(cluster_pos, axis=0)
            
            # Bounding box
            min_coords = np.min(cluster_pos, axis=0)
            max_coords = np.max(cluster_pos, axis=0)
            dimensions = max_coords - min_coords
            volume = np.prod(np.maximum(dimensions, 0.01))
            
            # Velocity from doppler
            avg_doppler = np.mean(cluster_doppler)
            range_val = np.linalg.norm(centroid)
            if range_val > 0.01:
                vel_direction = centroid / range_val
                velocity = vel_direction * avg_doppler
            else:
                velocity = np.array([0.0, avg_doppler, 0.0])
            
            # Confidence from SNR
            avg_snr = np.mean(cluster_snr) if len(cluster_snr) > 0 else 0
            confidence = min(1.0, max(0.0, avg_snr / 20.0))
            
            # Radar coverage - which radars contributed to this cluster
            unique_radars = np.unique(cluster_radars)
            radar_point_counts = {int(r): int(np.sum(cluster_radars == r)) for r in unique_radars}
            
            clusters.append({
                'centroid': centroid,
                'velocity': velocity,
                'dimensions': dimensions,
                'volume': volume,
                'num_points': num_points,
                'avg_doppler': avg_doppler,
                'avg_snr': avg_snr,
                'confidence': confidence,
                'timestamp': timestamp,
                'num_radars': len(unique_radars),
                'radar_point_counts': radar_point_counts
            })
        
        return clusters
    
    def _filter_clusters(self, clusters):
        """Filter clusters by physical constraints"""
        filtered = []
        
        for c in clusters:
            height = c['dimensions'][2]
            if height < self.min_cluster_height or height > self.max_cluster_height:
                continue
            
            if c['volume'] < self.min_cluster_volume or c['volume'] > self.max_cluster_volume:
                continue
            
            # Range check
            if np.linalg.norm(c['centroid']) > 15.0:
                continue
            
            filtered.append(c)
        
        return filtered
    
    def _associate_tracks(self, clusters, current_time):
        """Associate clusters with existing tracks or create new ones"""
        if len(clusters) == 0:
            return []
        
        cluster_centroids = np.array([c['centroid'] for c in clusters])
        
        if len(self.active_tracks) > 0:
            track_ids = list(self.active_tracks.keys())
            
            # Use KF predicted positions for association
            track_centroids = np.array([
                self.active_tracks[tid]['kf'].get_position() 
                for tid in track_ids
            ])
            
            # Distance matrix
            dist_matrix = cdist(cluster_centroids, track_centroids)
            
            # Greedy assignment
            assigned_clusters = set()
            assigned_tracks = set()
            
            while True:
                min_val = np.inf
                best_c, best_t = -1, -1
                
                for c_idx in range(len(clusters)):
                    if c_idx in assigned_clusters:
                        continue
                    for t_idx, tid in enumerate(track_ids):
                        if tid in assigned_tracks:
                            continue
                        if dist_matrix[c_idx, t_idx] < min_val:
                            min_val = dist_matrix[c_idx, t_idx]
                            best_c, best_t = c_idx, tid
                
                if min_val > self.max_track_distance:
                    break
                
                # Update existing track with KF
                self._update_track(best_t, clusters[best_c], current_time)
                assigned_clusters.add(best_c)
                assigned_tracks.add(best_t)
            
            # Create new tracks for unassigned clusters
            for c_idx in range(len(clusters)):
                if c_idx not in assigned_clusters:
                    self._create_track(clusters[c_idx], current_time)
        else:
            # No existing tracks - create new ones
            for c in clusters:
                self._create_track(c, current_time)
        
        return self._get_track_output()
    
    def _predict_all_tracks(self, current_time):
        """Run KF prediction on all active tracks (when no measurement available)"""
        for tid, track in self.active_tracks.items():
            track['kf'].predict()
            # Update track state from KF
            kf_state = track['kf'].get_state()
            track['centroid'] = kf_state['position']
            track['velocity'] = kf_state['velocity']
            track['acceleration'] = kf_state['acceleration']
    
    def _create_track(self, cluster, timestamp):
        """Create new track from cluster with Kalman Filter"""
        tid = self.next_track_id
        self.next_track_id += 1
        
        # Initialize Kalman Filter with cluster centroid and velocity
        kf = KalmanFilterCA(
            initial_pos=cluster['centroid'],
            initial_vel=cluster['velocity'],
            sigma_j=self.sigma_j,
            meas_sigma_pos=self.meas_sigma_pos,
            meas_sigma_vel=self.meas_sigma_vel,
            meas_sigma_acc=self.meas_sigma_acc
        )
        
        self.active_tracks[tid] = {
            'tid': tid,
            'kf': kf,  # Kalman Filter instance
            'centroid': cluster['centroid'],
            'velocity': cluster['velocity'],
            'acceleration': np.zeros(3),  # Will be estimated by KF
            'dimensions': cluster['dimensions'],
            'num_points': cluster['num_points'],
            'confidence': cluster['confidence'],
            'avg_snr': cluster['avg_snr'],
            'last_seen': timestamp,
            'first_seen': timestamp,
            'num_updates': 1,
            'num_radars': cluster['num_radars'],
            'radar_point_counts': cluster['radar_point_counts']
        }
    
    def _update_track(self, tid, cluster, timestamp):
        """Update existing track with new cluster measurement using KF"""
        track = self.active_tracks[tid]
        
        # KF predict step
        track['kf'].predict()
        
        # KF update step with measurement
        track['kf'].update(
            measurement_pos=cluster['centroid'],
            measurement_vel=cluster['velocity']
        )
        
        # Get filtered state from KF
        kf_state = track['kf'].get_state()
        
        # Update track with KF-filtered state
        track['centroid'] = kf_state['position']
        track['velocity'] = kf_state['velocity']
        track['acceleration'] = kf_state['acceleration']
        
        # Other attributes (not KF filtered)
        alpha = 0.5
        track['dimensions'] = alpha * cluster['dimensions'] + (1 - alpha) * track['dimensions']
        track['confidence'] = alpha * cluster['confidence'] + (1 - alpha) * track['confidence']
        track['num_points'] = cluster['num_points']
        track['avg_snr'] = cluster['avg_snr']
        track['last_seen'] = timestamp
        track['num_updates'] += 1
        track['num_radars'] = cluster['num_radars']
        track['radar_point_counts'] = cluster['radar_point_counts']
    
    def _cleanup_old_tracks(self, current_time):
        """Remove stale tracks"""
        expired = [tid for tid, t in self.active_tracks.items() 
                   if current_time - t['last_seen'] > self.track_timeout]
        for tid in expired:
            del self.active_tracks[tid]
    
    def _get_track_output(self):
        """Convert active tracks to output format"""
        output = []
        
        for tid, track in self.active_tracks.items():
            c = track['centroid']
            v = track['velocity']
            a = track['acceleration']
            
            output.append({
                'tid': track['tid'],
                'posX': float(c[0]),
                'posY': float(c[1]),
                'posZ': float(c[2]),
                'velX': float(v[0]),
                'velY': float(v[1]),
                'velZ': float(v[2]),
                'accX': float(a[0]),
                'accY': float(a[1]),
                'accZ': float(a[2]),
                'confidence': float(track['confidence']),
                'num_points': int(track['num_points']),
                'num_updates': int(track['num_updates']),
                'num_radars': int(track['num_radars']),
                'radar_point_counts': track['radar_point_counts'],
                'dimensions': track['dimensions'].tolist() if hasattr(track['dimensions'], 'tolist') else track['dimensions']
            })
        
        return output
    
    def _log(self, msg):
        """Log with prefix"""
        print(f'[UnifiedClusterer] {msg}')
