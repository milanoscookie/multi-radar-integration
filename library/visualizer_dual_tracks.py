"""
Fuser for Dual Radar System - Track Fusion & Output
Collects tracks from both radars, fuses them, and outputs to Industrial Visualizer
"""

import socket
import struct
import time
from datetime import datetime
import numpy as np

from library.track_fusion_pkg import TrackFusion


class FuseDualRadar:
    def __init__(self, run_flag, radar_rd_queue_list, vis_queue, shared_param_dict, **kwargs_CFG):
        """
        Initialize visualizer for dual radar track fusion
        """
        self.run_flag = run_flag
        self.radar_rd_queue_list = radar_rd_queue_list
        self.vis_queue = vis_queue
        self.status = shared_param_dict['proc_status_dict']
        self.status['Module_VIS'] = True
        
        # Config
        self.vis_cfg = kwargs_CFG['VISUALIZER_CFG']
        self.radar_cfg_list = kwargs_CFG['RADAR_CFG_LIST']
        self.ind_vis_cfg = kwargs_CFG.get('INDUSTRIAL_VIS_CFG', {})
        
        # Track fusion
        self.track_fusion = TrackFusion(**kwargs_CFG)
        
        # Statistics
        self.frame_count = 0
        self.total_tracks_processed = 0
        
        self._log('Visualizer initialized for dual radar')


    def run(self):
        """
        Main loop - collect frames from both radars, fuse tracks, and Send to Visualizer Queue
        """
        self._log('Starting visualization loop...')
        
        radar_frames_buffer = {cfg['name']: None for cfg in self.radar_cfg_list}
        last_output_time = time.time()
        output_period = 1.0 / 20  # 20 FPS output
        
        while self.run_flag.value:
            try:
                # Collect frames from all radars
                for i, queue in enumerate(self.radar_rd_queue_list):
                    if not queue.empty():
                        frame = queue.get(block=False)
                        radar_name = frame['radar_name']
                        radar_frames_buffer[radar_name] = frame
                
                # Check if it's time to process and output
                current_time = time.time()
                if current_time - last_output_time >= output_period:
                    # Get all available frames
                    available_frames = [f for f in radar_frames_buffer.values() if f is not None]
                    
                    if available_frames:
                        # Fuse tracks from all radars
                        fused_tracks = self.track_fusion.fuse_tracks(available_frames)
                        
                        if fused_tracks:
                            # Output to Industrial Visualizer
                            self._output_to_industrial_vis(fused_tracks)
                            
                            self.vis_queue.put(fused_tracks)

                            # Print statistics
                            self.total_tracks_processed += len(fused_tracks)
                            self.frame_count += 1
                            
                            if self.frame_count % 100 == 0:
                                self._log(f'Frames: {self.frame_count}, '
                                         f'Tracks: {len(fused_tracks)}, '
                                         f'Total: {self.total_tracks_processed}')
                        
                        # Clear buffer (ready for next sync point)
                        # radar_frames_buffer = {cfg['name']: None for cfg in self.radar_cfg_list}
                    
                    last_output_time = current_time
                
                # Small delay to prevent CPU spinning
                time.sleep(0.001)
                
            except Exception as e:
                self._log(f'Error in main loop: {e}')
                time.sleep(0.01)

    def _output_to_industrial_vis(self, fused_tracks):
        """
        Send fused tracks to Industrial Visualizer via socket
        Format: TLV structure compatible with Industrial Visualizer
        """
        # if not self.socket_conn:
        #     return
        
        # try:
        #     # Build TLV packet for Industrial Visualizer
        #     # Header: magic word + version + length + frame_num + num_tlvs
        #     MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
            
        #     # Build TLV 1010 data (3D Target List)
        #     tlv_data = self._build_tlv_1010(fused_tracks)
            
        #     # TLV header: type (4 bytes) + length (4 bytes)
        #     tlv_type = struct.pack('I', 1010)
        #     tlv_length = struct.pack('I', len(tlv_data))
            
        #     # Frame header (40 bytes total)
        #     version = struct.pack('I', 0xA1642333)  # SDK version
        #     total_packet_len = struct.pack('I', 40 + 8 + len(tlv_data))  # header + tlv_header + data
        #     platform = struct.pack('I', 0xA1843)  # IWR6843
        #     frame_number = struct.pack('I', self.frame_count)
        #     time_cpu = struct.pack('I', int(time.time() * 1000))
        #     num_detected = struct.pack('I', len(fused_tracks))
        #     num_tlvs = struct.pack('I', 1)  # Only TLV 1010
        #     subframe = struct.pack('I', 0)
            
        #     # Assemble packet
        #     packet = (MAGIC_WORD + version + total_packet_len + platform + 
        #              frame_number + time_cpu + num_detected + num_tlvs + subframe +
        #              tlv_type + tlv_length + tlv_data)
            
        #     # Send via socket
        #     if self.ind_vis_cfg.get('output_protocol', 'udp') == 'udp':
        #         self.socket_conn.sendto(packet, self.socket_address)
        #     else:
        #         self.socket_conn.send(packet)
            
        # except Exception as e:
        #     self._log(f'Output error: {e}')

        # Clear screen for clean output (optional)
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("="*70)
        print(f"Frame {self.frame_count} | Time: {time.strftime('%H:%M:%S')}")
        print(f"Total Tracks: {len(fused_tracks)}")
        print("="*70)
        
        for track in fused_tracks:
            print(f"Track ID: {track['global_tid']:3d} | "
                f"Pos: ({track['posX']:6.2f}, {track['posY']:6.2f}, {track['posZ']:6.2f}) | "
                f"Vel: ({track['velX']:5.2f}, {track['velY']:5.2f}, {track['velZ']:5.2f}) | "
                f"Conf: {track['confidence']:.2f} | "
                f"Radars: {track.get('num_radars_detected', 1)}")
            
            # Show which radars detected this track
            if 'source_radars' in track:
                print(f"         └─ Detected by: {', '.join(track['source_radars'])}")
        
        print("="*70)
        print()

    def _build_tlv_1010(self, fused_tracks):
        """
        Build TLV 1010 binary data from fused tracks
        Each track: 112 bytes
        """
        tlv_data = b''
        
        for track in fused_tracks:
            # Pack each track (112 bytes)
            tid = struct.pack('I', track['global_tid'])
            
            posX = struct.pack('f', track['posX'])
            posY = struct.pack('f', track['posY'])
            posZ = struct.pack('f', track['posZ'])
            
            velX = struct.pack('f', track['velX'])
            velY = struct.pack('f', track['velY'])
            velZ = struct.pack('f', track['velZ'])
            
            accX = struct.pack('f', track['accX'])
            accY = struct.pack('f', track['accY'])
            accZ = struct.pack('f', track['accZ'])
            
            # Error covariance (16 floats = 64 bytes) - zeros for now
            ec = struct.pack('16f', *([0.0] * 16))
            
            # Gating gain
            g = struct.pack('f', track.get('gating_gain', 1.0))
            
            # Confidence
            confidence = struct.pack('f', track['confidence'])
            
            # Combine into single track
            track_data = (tid + posX + posY + posZ + 
                         velX + velY + velZ + 
                         accX + accY + accZ + 
                         ec + g + confidence)
            
            tlv_data += track_data
        
        return tlv_data

    def _log(self, txt):
        """Print with module name"""
        print(f'[Visualizer]\t{txt}')

    def __del__(self):
        """Cleanup on exit"""
    
        self._log(f'Closed. Frames: {self.frame_count}, '
                 f'Total tracks: {self.total_tracks_processed}')
        self._log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        self.status['Module_VIS'] = False


if __name__ == '__main__':
    # Test visualizer
    from multiprocessing import Manager
    
    run_flag = Manager().Value('b', True)
    queue1 = Manager().Queue()
    queue2 = Manager().Queue()
    shared_dict = {'proc_status_dict': Manager().dict()}
    
    # Test config
    kwargs_CFG = {
        'VISUALIZER_CFG': {
            'dimension': '3D',
            'VIS_xlim': (-5, 5),
            'VIS_ylim': (0, 10),
            'VIS_zlim': (0, 3)
        },
        'RADAR_CFG_LIST': [
            {'name': 'Radar1'},
            {'name': 'Radar2'}
        ],
        'TRACK_FUSION_CFG': {
            'distance_threshold': 0.5
        },
        'INDUSTRIAL_VIS_CFG': {
            'enable': True,
            'output_protocol': 'udp',
            'ip_address': '127.0.0.1',
            'port': 5000
        }
    }
    
    # vis = Visualizer(run_flag, [queue1, queue2], shared_dict, **kwargs_CFG)
    
    # Simulate some test data
    test_frame1 = {
        'radar_name': 'Radar1',
        'timestamp': time.time(),
        'num_tracks': 1,
        'tracks': [
            {'tid': 1, 'posX': 1.0, 'posY': 2.0, 'posZ': 1.5,
             'velX': 0.1, 'velY': 0.2, 'velZ': 0.0,
             'accX': 0.0, 'accY': 0.0, 'accZ': 0.0,
             'confidence': 0.9, 'gating_gain': 1.0}
        ]
    }
    
    queue1.put(test_frame1)
    
    print("Visualizer test - press Ctrl+C to stop")
    try:
        vis.run()
    except KeyboardInterrupt:
        print("\nStopping...")
        run_flag.value = False