"""
Track Fusion Visualizer - maintains track history for visualization.
"""


class TrackFusionVisualizer:
    """Maintains track history for trail visualization."""
    
    def __init__(self, max_history=50):
        self.track_history = {}
        self.max_history = max_history

    def update(self, fused_tracks):
        """
        Update track history with new fused tracks.
        
        Args:
            fused_tracks: list of fused track dicts with global_tid, posX, posY, posZ
        """
        for track in fused_tracks:
            tid = track['global_tid']
            pos = (track['posX'], track['posY'], track['posZ'])
            self.track_history.setdefault(tid, []).append(pos)
            if len(self.track_history[tid]) > self.max_history:
                self.track_history[tid].pop(0)

    def get_track_trails(self):
        """
        Get all track trails for visualization.
        
        Returns:
            dict of global_tid -> list of (x, y, z) positions
        """
        return self.track_history

    def clear(self):
        """Clear all track history."""
        self.track_history = {}

    def remove_track(self, global_tid):
        """Remove a specific track from history."""
        if global_tid in self.track_history:
            del self.track_history[global_tid]
