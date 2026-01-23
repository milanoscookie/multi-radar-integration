"""
Test script for DBSCAN Point Cloud Clustering

Tests the clustering process with synthetic point cloud data.
Uses direct method calls to test clustering logic without multiprocessing.
"""

import numpy as np
import time


class MockClusterer:
    """
    Test wrapper that replicates the clustering logic from PointCloudClustererProcess
    without the multiprocessing infrastructure.
    """
    
    def __init__(self, cfg):
        self.cfg = cfg.get('POINTCLOUD_CLUSTERING_CFG', {})
        
        # DBSCAN parameters
        self.eps = self.cfg.get('eps', 0.5)
        self.min_samples = self.cfg.get('min_samples', 3)
        self.min_points_per_cluster = self.cfg.get('min_points_per_cluster', 3)
        
        # Physical filtering
        self.min_cluster_height = self.cfg.get('min_cluster_height', 0.3)
        self.max_cluster_height = self.cfg.get('max_cluster_height', 2.5)
        self.min_cluster_volume = self.cfg.get('min_cluster_volume', 0.01)
        self.max_cluster_volume = self.cfg.get('max_cluster_volume', 5.0)
        
        # Tracking
        self.max_track_distance = self.cfg.get('max_track_distance', 1.5)
        self.track_timeout = self.cfg.get('track_timeout', 1.0)
        
        self.active_tracks = {}
        self.next_track_id = 1
        
        print('[MockClusterer] Initialized')
        print(f'  DBSCAN: eps={self.eps}m, min_samples={self.min_samples}')
    
    def process_frame(self, frame):
        """Process a point cloud frame and return tracked objects"""
        from sklearn.cluster import DBSCAN
        from scipy.spatial.distance import cdist
        
        current_time = frame.get('timestamp', time.time())
        point_cloud = frame.get('point_cloud', {})
        points = point_cloud.get('points', [])
        
        if len(points) < self.min_samples:
            self._cleanup_old_tracks(current_time)
            return []
        
        # Convert to numpy
        positions = np.array([[pt['x'], pt['y'], pt['z']] for pt in points])
        dopplers = np.array([pt['doppler'] for pt in points])
        snrs = np.array([pt.get('snr') or 0 for pt in points])
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(positions)
        
        # Extract clusters
        clusters = []
        for label in set(labels):
            if label == -1:
                continue
            
            mask = labels == label
            cluster_pos = positions[mask]
            cluster_doppler = dopplers[mask]
            cluster_snr = snrs[mask]
            
            num_points = len(cluster_pos)
            if num_points < self.min_points_per_cluster:
                continue
            
            centroid = np.mean(cluster_pos, axis=0)
            min_coords = np.min(cluster_pos, axis=0)
            max_coords = np.max(cluster_pos, axis=0)
            dimensions = max_coords - min_coords
            volume = np.prod(np.maximum(dimensions, 0.01))
            
            avg_doppler = np.mean(cluster_doppler)
            range_val = np.linalg.norm(centroid)
            if range_val > 0.01:
                velocity = (centroid / range_val) * avg_doppler
            else:
                velocity = np.array([0.0, avg_doppler, 0.0])
            
            avg_snr = np.mean(cluster_snr)
            confidence = min(1.0, max(0.0, avg_snr / 20.0))
            
            # Physical filtering
            height = dimensions[2]
            if height < self.min_cluster_height or height > self.max_cluster_height:
                continue
            if volume < self.min_cluster_volume or volume > self.max_cluster_volume:
                continue
            if range_val > 15.0:
                continue
            
            clusters.append({
                'centroid': centroid,
                'velocity': velocity,
                'dimensions': dimensions,
                'volume': volume,
                'num_points': num_points,
                'avg_snr': avg_snr,
                'confidence': confidence,
                'timestamp': current_time
            })
        
        # Associate with tracks
        tracked_objects = self._associate_tracks(clusters, current_time)
        self._cleanup_old_tracks(current_time)
        
        return tracked_objects
    
    def _associate_tracks(self, clusters, current_time):
        from scipy.spatial.distance import cdist
        
        if len(clusters) == 0:
            return []
        
        cluster_centroids = np.array([c['centroid'] for c in clusters])
        
        if len(self.active_tracks) > 0:
            track_ids = list(self.active_tracks.keys())
            track_centroids = np.array([self.active_tracks[tid]['centroid'] for tid in track_ids])
            
            dist_matrix = cdist(cluster_centroids, track_centroids)
            
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
                
                # Update track
                self._update_track(best_t, clusters[best_c], current_time)
                assigned_clusters.add(best_c)
                assigned_tracks.add(best_t)
            
            # Create new tracks
            for c_idx in range(len(clusters)):
                if c_idx not in assigned_clusters:
                    self._create_track(clusters[c_idx], current_time)
        else:
            for c in clusters:
                self._create_track(c, current_time)
        
        return self._get_output()
    
    def _create_track(self, cluster, timestamp):
        tid = self.next_track_id
        self.next_track_id += 1
        self.active_tracks[tid] = {
            'tid': tid,
            'centroid': cluster['centroid'],
            'velocity': cluster['velocity'],
            'dimensions': cluster['dimensions'],
            'num_points': cluster['num_points'],
            'confidence': cluster['confidence'],
            'last_seen': timestamp,
            'num_updates': 1
        }
    
    def _update_track(self, tid, cluster, timestamp):
        track = self.active_tracks[tid]
        alpha = 0.5
        track['centroid'] = alpha * cluster['centroid'] + (1 - alpha) * track['centroid']
        track['velocity'] = alpha * cluster['velocity'] + (1 - alpha) * track['velocity']
        track['dimensions'] = alpha * cluster['dimensions'] + (1 - alpha) * track['dimensions']
        track['confidence'] = alpha * cluster['confidence'] + (1 - alpha) * track['confidence']
        track['num_points'] = cluster['num_points']
        track['last_seen'] = timestamp
        track['num_updates'] += 1
    
    def _cleanup_old_tracks(self, current_time):
        expired = [tid for tid, t in self.active_tracks.items() 
                   if current_time - t['last_seen'] > self.track_timeout]
        for tid in expired:
            del self.active_tracks[tid]
    
    def _get_output(self):
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
                'accX': 0.0, 'accY': 0.0, 'accZ': 0.0,
                'confidence': float(track['confidence']),
                'num_points': int(track['num_points']),
                'num_updates': int(track['num_updates']),
                'dimensions': track['dimensions'].tolist()
            })
        return output


# Test configuration
test_cfg = {
    'POINTCLOUD_CLUSTERING_CFG': {
        'eps': 0.5,
        'min_samples': 3,
        'min_points_per_cluster': 3,
        'min_cluster_height': 0.3,
        'max_cluster_height': 2.5,
        'min_cluster_volume': 0.01,
        'max_cluster_volume': 5.0,
        'max_track_distance': 1.5,
        'track_timeout': 1.0,
    }
}


def generate_synthetic_person(center, num_points=50, noise=0.05):
    """Generate synthetic person-like point cloud"""
    points = []
    for _ in range(num_points):
        dx = np.random.uniform(-0.3, 0.3)
        dy = np.random.uniform(-0.2, 0.2)
        dz = np.random.uniform(0, 1.7)
        dx += np.random.normal(0, noise)
        dy += np.random.normal(0, noise)
        dz += np.random.normal(0, noise)
        
        velocity_y = 0.5
        range_xy = np.sqrt((center[0] + dx)**2 + (center[1] + dy)**2)
        doppler = velocity_y * (center[1] + dy) / range_xy if range_xy > 0 else 0
        
        points.append({
            'x': center[0] + dx,
            'y': center[1] + dy,
            'z': center[2] + dz,
            'doppler': doppler,
            'snr': np.random.uniform(10, 20),
            'noise': np.random.uniform(2, 5)
        })
    return points


def test_single_person():
    print("\n" + "="*80)
    print("TEST 1: Single Person Cluster")
    print("="*80)
    
    clusterer = MockClusterer(test_cfg)
    points = generate_synthetic_person(center=(1.0, 2.0, 0.0), num_points=50)
    
    frame = {
        'radar_name': 'TestRadar',
        'point_cloud': {'points': points, 'coordinate_type': 'cartesian'},
        'timestamp': time.time()
    }
    
    tracked_objects = clusterer.process_frame(frame)
    
    print(f"  Input points: {len(points)}")
    print(f"  Detected objects: {len(tracked_objects)}")
    
    if len(tracked_objects) > 0:
        obj = tracked_objects[0]
        print(f"\n  Object 0:")
        print(f"    TID: {obj['tid']}")
        print(f"    Position: ({obj['posX']:.2f}, {obj['posY']:.2f}, {obj['posZ']:.2f})")
        print(f"    Num points: {obj['num_points']}")
        
        expected_pos = np.array([1.0, 2.0, 0.85])
        detected_pos = np.array([obj['posX'], obj['posY'], obj['posZ']])
        error = np.linalg.norm(detected_pos - expected_pos)
        print(f"    Position error: {error:.3f} m")
        
        if error < 0.5:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL: Position error too large")
    else:
        print("  ✗ FAIL: No objects detected")


def test_two_people():
    print("\n" + "="*80)
    print("TEST 2: Two People (Separated)")
    print("="*80)
    
    clusterer = MockClusterer(test_cfg)
    
    points1 = generate_synthetic_person(center=(1.0, 2.0, 0.0), num_points=40)
    points2 = generate_synthetic_person(center=(3.0, 3.0, 0.0), num_points=40)
    points = points1 + points2
    
    frame = {
        'radar_name': 'TestRadar',
        'point_cloud': {'points': points, 'coordinate_type': 'cartesian'},
        'timestamp': time.time()
    }
    
    tracked_objects = clusterer.process_frame(frame)
    
    print(f"  Input points: {len(points)}")
    print(f"  Detected objects: {len(tracked_objects)}")
    
    if len(tracked_objects) == 2:
        print("  ✓ PASS: Detected 2 objects")
    else:
        print(f"  ✗ FAIL: Expected 2 objects, got {len(tracked_objects)}")


def test_temporal_tracking():
    print("\n" + "="*80)
    print("TEST 3: Temporal Tracking (Moving Person)")
    print("="*80)
    
    clusterer = MockClusterer(test_cfg)
    positions = [(1.0, 2.0, 0.0), (1.0, 2.5, 0.0), (1.0, 3.0, 0.0), (1.0, 3.5, 0.0)]
    track_ids = []
    
    for frame_idx, pos in enumerate(positions):
        points = generate_synthetic_person(center=pos, num_points=50)
        frame = {
            'radar_name': 'TestRadar',
            'point_cloud': {'points': points, 'coordinate_type': 'cartesian'},
            'timestamp': time.time()
        }
        
        tracked_objects = clusterer.process_frame(frame)
        
        print(f"  Frame {frame_idx}: pos={pos}, detected={len(tracked_objects)}")
        if len(tracked_objects) > 0:
            track_ids.append(tracked_objects[0]['tid'])
            print(f"    TID: {tracked_objects[0]['tid']}, updates: {tracked_objects[0]['num_updates']}")
        
        time.sleep(0.1)
    
    if len(set(track_ids)) == 1:
        print("\n  ✓ PASS: Same track ID maintained")
    else:
        print(f"\n  ✗ FAIL: Track ID changed: {track_ids}")


def test_track_timeout():
    print("\n" + "="*80)
    print("TEST 4: Track Timeout")
    print("="*80)
    
    clusterer = MockClusterer(test_cfg)
    
    points = generate_synthetic_person(center=(1.0, 2.0, 0.0), num_points=50)
    frame = {
        'radar_name': 'TestRadar',
        'point_cloud': {'points': points, 'coordinate_type': 'cartesian'},
        'timestamp': time.time()
    }
    tracked_objects = clusterer.process_frame(frame)
    print(f"  Frame 1: Detected {len(tracked_objects)} object(s)")
    
    time.sleep(0.5)
    frame_empty = {
        'radar_name': 'TestRadar',
        'point_cloud': {'points': [], 'coordinate_type': 'cartesian'},
        'timestamp': time.time()
    }
    tracked_objects = clusterer.process_frame(frame_empty)
    print(f"  Frame 2 (empty): Active tracks = {len(clusterer.active_tracks)}")
    
    time.sleep(1.5)
    tracked_objects = clusterer.process_frame(frame_empty)
    print(f"  Frame 3 (after timeout): Active tracks = {len(clusterer.active_tracks)}")
    
    if len(clusterer.active_tracks) == 0:
        print("  ✓ PASS: Track timed out")
    else:
        print("  ✗ FAIL: Track did not timeout")


def test_noise_rejection():
    print("\n" + "="*80)
    print("TEST 5: Noise Point Rejection")
    print("="*80)
    
    clusterer = MockClusterer(test_cfg)
    
    person_points = generate_synthetic_person(center=(2.0, 3.0, 0.0), num_points=50)
    
    noise_points = []
    for _ in range(20):
        noise_points.append({
            'x': np.random.uniform(-3, 3),
            'y': np.random.uniform(0, 5),
            'z': np.random.uniform(0, 2),
            'doppler': np.random.uniform(-2, 2),
            'snr': np.random.uniform(5, 10),
            'noise': np.random.uniform(5, 10)
        })
    
    all_points = person_points + noise_points
    
    frame = {
        'radar_name': 'TestRadar',
        'point_cloud': {'points': all_points, 'coordinate_type': 'cartesian'},
        'timestamp': time.time()
    }
    
    tracked_objects = clusterer.process_frame(frame)
    
    print(f"  Input: {len(all_points)} points (50 person + 20 noise)")
    print(f"  Detected objects: {len(tracked_objects)}")
    
    if len(tracked_objects) == 1:
        print("  ✓ PASS: Noise rejected")
    else:
        print(f"  ⚠ WARNING: Expected 1, got {len(tracked_objects)}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("DBSCAN POINT CLOUD CLUSTERING TEST SUITE")
    print("="*80)
    
    test_single_person()
    test_two_people()
    test_temporal_tracking()
    test_track_timeout()
    test_noise_rejection()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
