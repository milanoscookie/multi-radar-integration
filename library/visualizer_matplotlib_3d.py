"""
Real-time 3D Matplotlib Visualizer (No UDP needed)
Displays fused tracks in 3D plot with track trails
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from library.track_fusion_pkg import TrackFusion


class MatplotlibVisualizer:
    def __init__(self, run_flag, radar_rd_queue_list, shared_param_dict, **kwargs_CFG):
        """
        Initialize matplotlib 3D visualizer
        """
        self.run_flag = run_flag
        self.radar_rd_queue_list = radar_rd_queue_list
        self.status = shared_param_dict['proc_status_dict']
        self.status['Module_VIS'] = True
        
        # Config
        self.vis_cfg = kwargs_CFG['VISUALIZER_CFG']
        self.radar_cfg_list = kwargs_CFG['RADAR_CFG_LIST']
        
        # Track fusion
        self.track_fusion = TrackFusion(**kwargs_CFG)
        
        # Visualization limits
        self.xlim = self.vis_cfg.get('VIS_xlim', (-5, 5))
        self.ylim = self.vis_cfg.get('VIS_ylim', (0, 10))
        self.zlim = self.vis_cfg.get('VIS_zlim', (0, 3))
        
        # Data storage
        self.current_tracks = []
        self.track_history = {}  # global_tid -> list of positions
        self.max_history = 50
        
        # Statistics
        self.frame_count = 0
        self.total_tracks_processed = 0
        
        # Setup matplotlib
        self.fig = None
        self.ax = None
        self.setup_plot()
        
        self._log('Matplotlib 3D Visualizer initialized')

    def setup_plot(self):
        """Setup 3D matplotlib figure"""
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set labels
        self.ax.set_xlabel('X (meters)', fontsize=10)
        self.ax.set_ylabel('Y (meters)', fontsize=10)
        self.ax.set_zlabel('Z (meters)', fontsize=10)
        self.ax.set_title('Dual Radar 3D Track Fusion', fontsize=14, fontweight='bold')
        
        # Set limits
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_zlim(self.zlim)
        
        # Grid
        self.ax.grid(True, alpha=0.3)
        
        # View angle
        self.ax.view_init(elev=20, azim=45)
        
        # Add ground plane
        xx, yy = np.meshgrid(
            np.linspace(self.xlim[0], self.xlim[1], 10),
            np.linspace(self.ylim[0], self.ylim[1], 10)
        )
        zz = np.zeros_like(xx)
        self.ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
        
        # Add radar positions (if configured)
        self.plot_radar_positions()

    def plot_radar_positions(self):
        """Plot radar locations"""
        for radar_cfg in self.radar_cfg_list:
            pos = radar_cfg.get('pos_offset', (0, 0, 0))
            self.ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                          c='red', marker='^', s=200, 
                          label=f"{radar_cfg['name']}", alpha=0.7)
        self.ax.legend(loc='upper right')

    def run(self):
        """
        Main loop - get frames, fuse, and update plot
        """
        self._log('Starting visualization loop...')
        
        radar_frames_buffer = {cfg['name']: None for cfg in self.radar_cfg_list}
        last_output_time = time.time()
        output_period = 1.0 / 20  # 20 FPS
        
        try:
            while self.run_flag.value:
                # Collect frames from all radars
                for i, queue in enumerate(self.radar_rd_queue_list):
                    if not queue.empty():
                        frame = queue.get(block=False)
                        radar_name = frame['radar_name']
                        radar_frames_buffer[radar_name] = frame
                
                # Check if it's time to update
                current_time = time.time()
                if current_time - last_output_time >= output_period:
                    available_frames = [f for f in radar_frames_buffer.values() if f is not None]
                    
                    if available_frames:
                        # Fuse tracks
                        fused_tracks = self.track_fusion.fuse_tracks(available_frames)
                        
                        if fused_tracks:
                            # Update visualization
                            self.current_tracks = fused_tracks
                            self.update_track_history(fused_tracks)
                            self.update_plot()
                            
                            # Statistics
                            self.total_tracks_processed += len(fused_tracks)
                            self.frame_count += 1
                            
                            if self.frame_count % 100 == 0:
                                self._log(f'Frames: {self.frame_count}, '
                                         f'Tracks: {len(fused_tracks)}, '
                                         f'Total: {self.total_tracks_processed}')
                    
                    last_output_time = current_time
                
                # Small delay
                plt.pause(0.001)
                
        except KeyboardInterrupt:
            self._log('Interrupted by user')
        except Exception as e:
            self._log(f'Error: {e}')
            import traceback
            traceback.print_exc()
        finally:
            plt.close(self.fig)

    def update_track_history(self, fused_tracks):
        """Update track position history for trails"""
        current_tids = set()
        
        for track in fused_tracks:
            tid = track['global_tid']
            current_tids.add(tid)
            pos = (track['posX'], track['posY'], track['posZ'])
            
            if tid not in self.track_history:
                self.track_history[tid] = []
            
            self.track_history[tid].append(pos)
            
            # Keep only recent history
            if len(self.track_history[tid]) > self.max_history:
                self.track_history[tid].pop(0)
        
        # Remove old tracks (not seen in last 2 seconds)
        old_tids = set(self.track_history.keys()) - current_tids
        for tid in old_tids:
            if len(self.track_history[tid]) > 10:  # Keep some history
                self.track_history[tid].pop(0)
            else:
                del self.track_history[tid]

    def update_plot(self):
        """Update 3D plot with current tracks"""
        # Clear previous data
        self.ax.clear()
        
        # Reset plot settings
        self.ax.set_xlabel('X (meters)', fontsize=10)
        self.ax.set_ylabel('Y (meters)', fontsize=10)
        self.ax.set_zlabel('Z (meters)', fontsize=10)
        self.ax.set_title(f'Dual Radar 3D Track Fusion | Frame {self.frame_count}', 
                         fontsize=14, fontweight='bold')
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_zlim(self.zlim)
        self.ax.grid(True, alpha=0.3)
        
        # Ground plane
        xx, yy = np.meshgrid(
            np.linspace(self.xlim[0], self.xlim[1], 10),
            np.linspace(self.ylim[0], self.ylim[1], 10)
        )
        zz = np.zeros_like(xx)
        self.ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
        
        # Plot radar positions
        self.plot_radar_positions()
        
        # Plot track trails
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.track_history)))
        for i, (tid, history) in enumerate(self.track_history.items()):
            if len(history) > 1:
                history_array = np.array(history)
                self.ax.plot(history_array[:, 0], 
                           history_array[:, 1], 
                           history_array[:, 2],
                           color=colors[i], alpha=0.5, linewidth=2)
        
        # Plot current tracks
        for track in self.current_tracks:
            x, y, z = track['posX'], track['posY'], track['posZ']
            tid = track['global_tid']
            num_radars = track.get('num_radars_detected', 1)
            confidence = track['confidence']
            
            # Color based on number of radars
            if num_radars >= 2:
                color = 'green'  # Detected by multiple radars
                marker = 'o'
                size = 200
            else:
                color = 'blue'  # Detected by single radar
                marker = 'o'
                size = 100
            
            # Plot track position
            self.ax.scatter([x], [y], [z], c=color, marker=marker, 
                          s=size, alpha=0.8, edgecolors='black', linewidths=2)
            
            # Add label
            label = f"ID:{tid}\nC:{confidence:.2f}\nR:{num_radars}"
            self.ax.text(x, y, z + 0.2, label, fontsize=8, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Draw velocity vector
            vx, vy, vz = track['velX'], track['velY'], track['velZ']
            if np.sqrt(vx**2 + vy**2 + vz**2) > 0.05:  # Only if moving
                self.ax.quiver(x, y, z, vx, vy, vz, 
                             length=1.0, color='red', arrow_length_ratio=0.3, linewidth=2)
        
        # Add info text
        info_text = (f"Tracks: {len(self.current_tracks)} | "
                    f"Frame: {self.frame_count} | "
                    f"Total: {self.total_tracks_processed}")
        self.ax.text2D(0.05, 0.95, info_text, transform=self.ax.transAxes,
                      fontsize=12, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Multi-Radar'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=10, label='Single Radar'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='Velocity')
        ]
        self.ax.legend(handles=legend_elements, loc='upper left')
        
        # Draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _log(self, txt):
        """Print with module name"""
        print(f'[MatplotlibVis]\t{txt}')

    def __del__(self):
        """Cleanup on exit"""
        self._log(f'Closed. Frames: {self.frame_count}, '
                 f'Total tracks: {self.total_tracks_processed}')
        self.status['Module_VIS'] = False
        plt.close('all')