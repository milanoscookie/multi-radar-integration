"""
Track ID Management for multi-radar fusion.
Handles global TID assignment, mapping, and cleanup.
"""

import time


class TrackIDManager:
    """Manages global track ID assignment and lifecycle."""
    
    def __init__(self, tid_timeout=2.0):
        self.next_global_tid = 1
        self.radar_to_global_tid = {}      # (radar_name, local_tid) -> global_tid
        self.global_tid_last_seen = {}     # global_tid -> time.time()
        self.tid_timeout = float(tid_timeout)

    def get_global_tid(self, radar_name, local_tid):
        """Get existing global TID for a radar/local pair, or None if not mapped."""
        key = (radar_name, local_tid)
        return self.radar_to_global_tid.get(key)

    def is_established(self, radar_name, local_tid):
        """Check if a radar/local pair has an established global TID."""
        key = (radar_name, local_tid)
        return key in self.radar_to_global_tid

    def assign_global_tid(self, track):
        """
        Assign a global TID to a single track.
        If the radar/local pair already has a mapping, use it.
        Otherwise, create a new global TID.
        
        Returns the track dict with global_tid added.
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

        self.global_tid_last_seen[global_tid] = time.time()

        track['global_tid'] = global_tid
        track['num_radars_detected'] = 1
        track['source_radars'] = [radar_name]
        track['source_tids'] = [key]
        return track

    def assign_global_tid_multi(self, source_tracks, fused_track):
        """
        Assign a global TID to a cluster of tracks from multiple radars.
        Uses the oldest existing TID if any track has one, otherwise creates new.
        
        Updates all source tracks to map to the same global TID.
        Returns the fused_track dict with global_tid added.
        """
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

        # Update all source tracks to point to this global TID
        for t in source_tracks:
            self.radar_to_global_tid[(t['radar_name'], t['tid'])] = global_tid

        self.global_tid_last_seen[global_tid] = time.time()
        fused_track['global_tid'] = global_tid
        return fused_track

    def update_last_seen(self, global_tid):
        """Update the last-seen timestamp for a global TID."""
        self.global_tid_last_seen[global_tid] = time.time()

    def remap_tid(self, radar_name, local_tid, new_global_tid):
        """Remap a radar/local pair to a new global TID."""
        key = (radar_name, local_tid)
        self.radar_to_global_tid[key] = new_global_tid

    def remove_mapping(self, radar_name, local_tid):
        """Remove the mapping for a radar/local pair."""
        key = (radar_name, local_tid)
        if key in self.radar_to_global_tid:
            del self.radar_to_global_tid[key]

    def get_mappings_for_gid(self, global_tid):
        """Get all (radar_name, local_tid) keys that map to a global TID."""
        return [k for k, v in self.radar_to_global_tid.items() if v == global_tid]

    def cleanup_old_tids(self, filters_dict, global_tid_last_ts):
        """
        Remove global TIDs that haven't been seen recently.
        Also cleans up associated EKF filters and timestamps.
        
        Args:
            filters_dict: dict of global_tid -> EKFTrack to clean up
            global_tid_last_ts: dict of global_tid -> last measurement timestamp
            
        Returns:
            List of removed TIDs
        """
        now = time.time()
        old = [tid for tid, last in self.global_tid_last_seen.items()
               if now - last > self.tid_timeout]

        for tid in old:
            del self.global_tid_last_seen[tid]

            # Remove all radar->global mappings for this TID
            keys_to_remove = [k for k, v in self.radar_to_global_tid.items() if v == tid]
            for k in keys_to_remove:
                del self.radar_to_global_tid[k]

            # Clean up EKF state
            if tid in filters_dict:
                del filters_dict[tid]
            if tid in global_tid_last_ts:
                del global_tid_last_ts[tid]

        return old

    def merge_tids(self, keep_gid, remove_gid, filters_dict, global_tid_last_ts):
        """
        Merge remove_gid into keep_gid.
        Updates all mappings and cleans up the removed TID.
        
        Args:
            keep_gid: Global TID to keep
            remove_gid: Global TID to remove
            filters_dict: dict of global_tid -> EKFTrack
            global_tid_last_ts: dict of global_tid -> timestamp
        """
        # Update all radar->global mappings that pointed to remove_gid
        keys_to_update = [k for k, v in self.radar_to_global_tid.items() if v == remove_gid]
        for key in keys_to_update:
            self.radar_to_global_tid[key] = keep_gid

        # Clean up remove_gid from tracking dicts
        if remove_gid in self.global_tid_last_seen:
            del self.global_tid_last_seen[remove_gid]
        if remove_gid in global_tid_last_ts:
            del global_tid_last_ts[remove_gid]
        if remove_gid in filters_dict:
            del filters_dict[remove_gid]

        return keys_to_update
