"""
Point Cloud Clustering Process using DBSCAN
Sits between RadarReader (point cloud) and TrackFusion

Pipeline: RadarReader → PointCloudClustererProcess → TrackFusion

Input Queue: Point cloud frames from radar reader
Output Queue: Clustered tracked objects (compatible with track fusion)
"""

import time
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


class PointCloudClustererProcess:
    """
    Process that consumes point cloud frames and produces tracked object clusters.
    Runs as a separate process in the multiprocessing pipeline.
    """
    
    def __init__(self, run_flag, input_queue, output_queue, shared_param_dict, **kwargs_CFG):
        """
        Initialize clusterer process
        
        Args:
            run_flag: multiprocessing.Value for run control
            input_queue: Queue receiving point cloud frames from radar reader
            output_queue: Queue outputting clustered objects to track fusion
            shared_param_dict: Shared status dictionary
            kwargs_CFG: Configuration dict with POINTCLOUD_CLUSTERING_CFG and RADAR_CFG
        """
        self.run_flag = run_flag
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.status = shared_param_dict['proc_status_dict']
        
        # Get radar name for logging
        radar_cfg = kwargs_CFG.get('RADAR_CFG', {})
        self.radar_name = radar_cfg.get('name', 'Unknown')
        
        self.status[f'Clusterer_{self.radar_name}'] = True
        
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
        
        # Track state
        self.active_tracks = {}
        self.next_track_id = 1
        
        # Statistics
        self.frame_count = 0
        self.total_clusters = 0
        self.total_points = 0
        
        self._log(f'Initialized for {self.radar_name}')
        self._log(f'  DBSCAN: eps={self.eps}m, min_samples={self.min_samples}')
    
    def run(self):
        """Main process loop - consume point clouds, produce tracked objects"""
        self._log('Starting clustering loop...')
        
        while self.run_flag.value:
            try:
                # Get point cloud frame from input queue (non-blocking with timeout)
                if not self.input_queue.empty():
                    frame = self.input_queue.get(block=False)
                    
                    # Process the frame
                    tracked_objects = self._process_frame(frame)
                    
                    # Output clustered objects if any found
                    if len(tracked_objects) > 0:
                        output_frame = {
                            'radar_name': frame.get('radar_name', self.radar_name),
                            'frame_number': frame.get('frame_number', self.frame_count),
                            'num_detected_obj': len(tracked_objects),
                            'tracks': tracked_objects,
                            'timestamp': frame.get('timestamp', time.time())
                        }
                        self.output_queue.put(output_frame)
                    
                    self.frame_count += 1
                    
                    if self.frame_count % 100 == 0:
                        avg_clusters = self.total_clusters / max(1, self.frame_count)
                        avg_points = self.total_points / max(1, self.frame_count)
                        self._log(f'Frames: {self.frame_count}, '
                                 f'Avg pts: {avg_points:.1f}, '
                                 f'Avg clusters: {avg_clusters:.1f}, '
                                 f'Active tracks: {len(self.active_tracks)}')
                else:
                    # No data available, small sleep
                    time.sleep(0.001)
                    
            except Exception as e:
                self._log(f'Error: {e}')
                time.sleep(0.01)
        
        self._log('Stopped')
        self.status[f'Clusterer_{self.radar_name}'] = False
    
    def _process_frame(self, frame):
        """
        Process a single point cloud frame
        
        Args:
            frame: dict with 'point_cloud' containing 'points' list
        
        Returns:
            List of tracked object dicts
        """
        current_time = frame.get('timestamp', time.time())
        point_cloud = frame.get('point_cloud', {})
        points = point_cloud.get('points', [])
        
        self.total_points += len(points)
        
        if len(points) < self.min_samples:
            self._cleanup_old_tracks(current_time)
            return []
        
        # Convert to numpy arrays
        positions = np.array([[pt['x'], pt['y'], pt['z']] for pt in points])
        dopplers = np.array([pt['doppler'] for pt in points])
        snrs = np.array([pt.get('snr') or 0 for pt in points])
        
        # Run DBSCAN
        cluster_labels = self._cluster_dbscan(positions)
        
        # Extract cluster features
        cluster_objects = self._extract_clusters(
            cluster_labels, positions, dopplers, snrs, current_time
        )
        
        # Filter invalid clusters
        cluster_objects = self._filter_clusters(cluster_objects)
        
        # Associate with existing tracks
        tracked_objects = self._associate_tracks(cluster_objects, current_time)
        
        # Cleanup stale tracks
        self._cleanup_old_tracks(current_time)
        
        self.total_clusters += len(tracked_objects)
        
        return tracked_objects
    
    def _cluster_dbscan(self, positions):
        """Apply DBSCAN clustering"""
        if len(positions) < self.min_samples:
            return np.full(len(positions), -1)
        
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=1)
        return dbscan.fit_predict(positions)
    
    def _extract_clusters(self, labels, positions, dopplers, snrs, timestamp):
        """Extract features for each cluster"""
        unique_labels = set(labels)
        clusters = []
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            mask = labels == label
            cluster_pos = positions[mask]
            cluster_doppler = dopplers[mask]
            cluster_snr = snrs[mask]
            
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
            
            # Velocity from doppler (radial direction)
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
            
            clusters.append({
                'centroid': centroid,
                'velocity': velocity,
                'dimensions': dimensions,
                'volume': volume,
                'num_points': num_points,
                'avg_doppler': avg_doppler,
                'avg_snr': avg_snr,
                'confidence': confidence,
                'timestamp': timestamp
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
            track_centroids = np.array([self.active_tracks[tid]['centroid'] for tid in track_ids])
            
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
                
                # Update existing track
                self._update_track(best_t, clusters[best_c], current_time)
                assigned_clusters.add(best_c)
                assigned_tracks.add(best_t)
            
            # Create new tracks for unassigned clusters
            for c_idx in range(len(clusters)):
                if c_idx not in assigned_clusters:
                    self._create_track(clusters[c_idx], current_time)
        else:
            # No existing tracks
            for c in clusters:
                self._create_track(c, current_time)
        
        return self._get_track_output()
    
    def _create_track(self, cluster, timestamp):
        """Create new track"""
        tid = self.next_track_id
        self.next_track_id += 1
        
        self.active_tracks[tid] = {
            'tid': tid,
            'centroid': cluster['centroid'],
            'velocity': cluster['velocity'],
            'dimensions': cluster['dimensions'],
            'num_points': cluster['num_points'],
            'confidence': cluster['confidence'],
            'avg_snr': cluster['avg_snr'],
            'last_seen': timestamp,
            'num_updates': 1
        }
    
    def _update_track(self, tid, cluster, timestamp):
        """Update existing track with new measurement"""
        track = self.active_tracks[tid]
        alpha = 0.5  # EMA weight
        
        track['centroid'] = alpha * cluster['centroid'] + (1 - alpha) * track['centroid']
        track['velocity'] = alpha * cluster['velocity'] + (1 - alpha) * track['velocity']
        track['dimensions'] = alpha * cluster['dimensions'] + (1 - alpha) * track['dimensions']
        track['confidence'] = alpha * cluster['confidence'] + (1 - alpha) * track['confidence']
        track['num_points'] = cluster['num_points']
        track['avg_snr'] = cluster['avg_snr']
        track['last_seen'] = timestamp
        track['num_updates'] += 1
    
    def _cleanup_old_tracks(self, current_time):
        """Remove stale tracks"""
        expired = [tid for tid, t in self.active_tracks.items() 
                   if current_time - t['last_seen'] > self.track_timeout]
        for tid in expired:
            del self.active_tracks[tid]
    
    def _get_track_output(self):
        """Convert tracks to output format compatible with track fusion"""
        output = []
        
        for tid, track in self.active_tracks.items():
            c = track['centroid']
            v = track['velocity']
            
            output.append({
                'tid': track['tid'],
                'posX': float(c[0]),
                'posY': float(c[1]),
                'posZ': float(c[2]),
                'velX': float(v[0]),
                'velY': float(v[1]),
                'velZ': float(v[2]),
                'accX': 0.0,
                'accY': 0.0,
                'accZ': 0.0,
                'confidence': float(track['confidence']),
                'num_points': int(track['num_points']),
                'num_updates': int(track['num_updates'])
            })
        
        return output
    
    def _log(self, msg):
        """Log with radar name prefix"""
        print(f'[Clusterer_{self.radar_name}] {msg}')
