"""
Track Fusion Module for Dual Radar System
Merges tracks from multiple radars into a single unified track list
"""

import numpy as np
from scipy.spatial.distance import cdist
import time


class TrackFusion:
    def __init__(self, **kwargs_CFG):
        """
        Initialize track fusion system
        """
        self.fusion_cfg = kwargs_CFG.get('TRACK_FUSION_CFG', {})
        self.radar_cfg_list = kwargs_CFG['RADAR_CFG_LIST']
        
        # Fusion parameters
        self.distance_threshold = self.fusion_cfg.get('distance_threshold', 0.5)  # meters
        self.confidence_weight = self.fusion_cfg.get('confidence_weight', 0.7)
        self.velocity_weight = self.fusion_cfg.get('velocity_weight', 0.3)
        self.fusion_method = self.fusion_cfg.get('fusion_method', 'weighted_average')
        
        # Track ID management
        self.next_global_tid = 1
        self.radar_to_global_tid = {}  # Maps (radar_name, local_tid) -> global_tid
        self.global_tid_last_seen = {}  # Tracks when global IDs were last seen
        self.tid_timeout = 2.0  # seconds
        
        self._log('Track Fusion initialized')

    def fuse_tracks(self, radar_frames):
        """
        Fuse tracks from multiple radar frames
        
        Args:
            radar_frames: List of frame dicts from different radars
                         Each dict has: {'radar_name', 'tracks', 'timestamp', ...}
        
        Returns:
            fused_tracks: List of fused track dictionaries
        """
        if not radar_frames:
            return []
        
        # Clean up old track IDs
        self._cleanup_old_tids()
        
        # Extract all tracks with radar info
        all_tracks = []
        for frame in radar_frames:
            radar_name = frame['radar_name']
            for track in frame['tracks']: # Each track is list of dicts -> transformed tracks
                track_copy = track.copy()
                track_copy['radar_name'] = radar_name
                track_copy['timestamp'] = frame['timestamp']
                all_tracks.append(track_copy)
            #     all_tracks = [
            #     {'tid': 5, 'posX': 1.0, 'posY': 2.0, 'posZ': 1.5, 'radar_name': 'Radar1'},
            #     {'tid': 7, 'posX': 3.0, 'posY': 4.0, 'posZ': 1.2, 'radar_name': 'Radar1'},
            #     {'tid': 3, 'posX': 1.1, 'posY': 2.1, 'posZ': 1.4, 'radar_name': 'Radar2'}]
        if not all_tracks:
            return []
        
        # Build distance matrix between all tracks
        positions = np.array([[t['posX'], t['posY'], t['posZ']] for t in all_tracks])
        dist_matrix = cdist(positions, positions, metric='euclidean')
        
        # Find tracks to merge (within distance threshold)
        fused_tracks = []
        merged_indices = set()
        
        for i, track_i in enumerate(all_tracks):
            if i in merged_indices:
                continue
            
            # Find all tracks close to track_i
            close_tracks_idx = np.where(dist_matrix[i, :] < self.distance_threshold)[0]
            close_tracks = [all_tracks[idx] for idx in close_tracks_idx if idx not in merged_indices]
            
            if len(close_tracks) > 1:
                # Multiple tracks from different radars - fuse them
                fused_track = self._merge_multiple_tracks(close_tracks)
                fused_tracks.append(fused_track)
                merged_indices.update(close_tracks_idx)
            else:
                # Single track - just assign global ID
                fused_track = self._assign_global_tid(track_i)
                fused_tracks.append(fused_track)
                merged_indices.add(i)
        
        return fused_tracks

    def _merge_multiple_tracks(self, tracks):
        """
        Merge multiple tracks from different radars into one
        Uses weighted average based on confidence
        """
        if len(tracks) == 1:
            return self._assign_global_tid(tracks[0])
        
        # Calculate weights based on confidence
        confidences = np.array([t.get('confidence', 0.5) for t in tracks])
        weights = confidences / np.sum(confidences)
        
        # Weighted average of positions
        positions = np.array([[t['posX'], t['posY'], t['posZ']] for t in tracks])
        fused_pos = np.average(positions, axis=0, weights=weights)
        
        # Weighted average of velocities
        velocities = np.array([[t['velX'], t['velY'], t['velZ']] for t in tracks])
        fused_vel = np.average(velocities, axis=0, weights=weights)
        
        # Weighted average of accelerations
        accelerations = np.array([[t['accX'], t['accY'], t['accZ']] for t in tracks])
        fused_acc = np.average(accelerations, axis=0, weights=weights)
        
        # Use maximum confidence
        fused_confidence = np.max(confidences)
        
        # Create fused track
        fused_track = {
            'posX': fused_pos[0],
            'posY': fused_pos[1],
            'posZ': fused_pos[2],
            'velX': fused_vel[0],
            'velY': fused_vel[1],
            'velZ': fused_vel[2],
            'accX': fused_acc[0],
            'accY': fused_acc[1],
            'accZ': fused_acc[2],
            'confidence': fused_confidence,
            'num_radars_detected': len(tracks),
            'source_radars': [t['radar_name'] for t in tracks],
            'source_tids': [(t['radar_name'], t['tid']) for t in tracks],
            'timestamp': np.mean([t['timestamp'] for t in tracks])
        }
        
        # Assign global track ID
        fused_track = self._assign_global_tid_multi(tracks, fused_track)
        
        return fused_track

    def _assign_global_tid(self, track):
        """
        Assign a global track ID to a single radar track
        Maintains consistency across frames
        """
        radar_name = track['radar_name']
        local_tid = track['tid']
        key = (radar_name, local_tid)
        
        if key in self.radar_to_global_tid:
            global_tid = self.radar_to_global_tid[key]
        else:
            global_tid = self.next_global_tid
            self.radar_to_global_tid[key] = global_tid
            self.next_global_tid += 1
        
        # Update last seen time
        self.global_tid_last_seen[global_tid] = time.time()
        
        track['global_tid'] = global_tid
        track['num_radars_detected'] = 1
        track['source_radars'] = [radar_name]
        track['source_tids'] = [key]
        
        return track

    def _assign_global_tid_multi(self, source_tracks, fused_track):
        """
        Assign global TID when fusing multiple tracks
        Tries to maintain existing global ID if any source track has one
        """
        # Check if any source track already has a global ID
        existing_global_tids = []
        for track in source_tracks:
            key = (track['radar_name'], track['tid'])
            if key in self.radar_to_global_tid:
                existing_global_tids.append(self.radar_to_global_tid[key])
        
        if existing_global_tids:
            # Use the most recent global TID
            global_tid = min(existing_global_tids)  # Use lowest (oldest) ID
        else:
            # Create new global TID
            global_tid = self.next_global_tid
            self.next_global_tid += 1
        
        # Update mappings for all source tracks
        for track in source_tracks:
            key = (track['radar_name'], track['tid'])
            self.radar_to_global_tid[key] = global_tid
        
        # Update last seen
        self.global_tid_last_seen[global_tid] = time.time()
        
        fused_track['global_tid'] = global_tid
        return fused_track

    def _cleanup_old_tids(self):
        """Remove track IDs that haven't been seen recently"""
        current_time = time.time()
        old_tids = [tid for tid, last_seen in self.global_tid_last_seen.items()
                    if current_time - last_seen > self.tid_timeout]
        
        for tid in old_tids:
            # Remove from last_seen
            del self.global_tid_last_seen[tid]
            
            # Remove from mapping
            keys_to_remove = [k for k, v in self.radar_to_global_tid.items() if v == tid]
            for key in keys_to_remove:
                del self.radar_to_global_tid[key]

    def _log(self, txt):
        print(f'[TrackFusion]\t{txt}')


class TrackFusionVisualizer:
    """
    Simple visualizer for fused tracks (optional, for debugging)
    """
    def __init__(self):
        self.track_history = {}  # global_tid -> list of positions
        self.max_history = 50
    
    def update(self, fused_tracks):
        """Update track history for visualization"""
        for track in fused_tracks:
            tid = track['global_tid']
            pos = (track['posX'], track['posY'], track['posZ'])
            
            if tid not in self.track_history:
                self.track_history[tid] = []
            
            self.track_history[tid].append(pos)
            
            # Keep only recent history
            if len(self.track_history[tid]) > self.max_history:
                self.track_history[tid].pop(0)
    
    def get_track_trails(self):
        """Get track trails for visualization"""
        return self.track_history


if __name__ == '__main__':
    # Test track fusion
    fusion_cfg = {
        'TRACK_FUSION_CFG': {
            'distance_threshold': 0.5,
            'confidence_weight': 0.7,
            'velocity_weight': 0.3
        },
        'RADAR_CFG_LIST': [
            {'name': 'Radar1'},
            {'name': 'Radar2'}
        ]
    }
    
    fusion = TrackFusion(**fusion_cfg)
    
    # Simulate two radar frames with overlapping tracks
    frame1 = {
        'radar_name': 'Radar1',
        'timestamp': time.time(),
        'tracks': [
            {'tid': 1, 'posX': 1.0, 'posY': 2.0, 'posZ': 1.5,
             'velX': 0.1, 'velY': 0.2, 'velZ': 0.0,
             'accX': 0.0, 'accY': 0.0, 'accZ': 0.0,
             'confidence': 0.9},
            {'tid': 2, 'posX': 3.0, 'posY': 4.0, 'posZ': 1.2,
             'velX': -0.1, 'velY': 0.0, 'velZ': 0.0,
             'accX': 0.0, 'accY': 0.0, 'accZ': 0.0,
             'confidence': 0.7}
        ]
    }
    
    frame2 = {
        'radar_name': 'Radar2',
        'timestamp': time.time(),
        'tracks': [
            {'tid': 5, 'posX': 1.1, 'posY': 2.1, 'posZ': 1.4,  # Close to Radar1 tid=1
             'velX': 0.15, 'velY': 0.18, 'velZ': 0.0,
             'accX': 0.0, 'accY': 0.0, 'accZ': 0.0,
             'confidence': 0.85}
        ]
    }
    
    fused = fusion.fuse_tracks([frame1, frame2])
    
    print(f'\nFused {len(fused)} tracks:')
    for track in fused:
        print(f"  Global TID {track['global_tid']}: "
              f"Pos=({track['posX']:.2f}, {track['posY']:.2f}, {track['posZ']:.2f}), "
              f"Conf={track['confidence']:.2f}, "
              f"Sources={track['source_radars']}")