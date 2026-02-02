"""
Point Cloud Radar with Unified DBSCAN Clustering

Simplified Pipeline (NO track fusion):
1. RadarReaderPointCloud (per radar) - Reads point cloud TLVs
2. UnifiedPointCloudClusterer - Merges all radars, clusters together

Pipeline:
[Radar 1] ──┐                    
            ├──> [Unified Clusterer] ──> [Tracked Objects] ──> [Visualizer]
[Radar 2] ──┘    (Merge + DBSCAN)

No track fusion needed - tracks are formed directly from merged point cloud!
"""

from multiprocessing import Process, Manager
from time import sleep

# Import the radar reader (point cloud)
from library.radar_reader_dual_pointcloud import RadarReaderPointCloud

# Import the unified clusterer (combines all radars)
from library.unified_pointcloud_clusterer import UnifiedPointCloudClusterer

# Import visualization
from library.sync_monitor import SyncMonitor

# Import configuration
from cfg.config_demo_dual_radar import *


def radar_pointcloud_proc(_run_flag, _radar_rd_queue, _shared_param_dict, **_kwargs_CFG):
    """
    Radar Reader Process
    Reads raw point cloud TLVs and outputs point cloud frames
    """
    radar = RadarReaderPointCloud(
        run_flag=_run_flag,
        radar_rd_queue=_radar_rd_queue,
        shared_param_dict=_shared_param_dict,
        **_kwargs_CFG
    )
    radar.run()


def unified_clusterer_proc(_run_flag, _input_queue_list, _output_queue, _shared_param_dict, **_kwargs_CFG):
    """
    Unified Clusterer Process
    Merges point clouds from all radars, clusters together to form tracks
    """
    clusterer = UnifiedPointCloudClusterer(
        run_flag=_run_flag,
        input_queue_list=_input_queue_list,
        output_queue=_output_queue,
        shared_param_dict=_shared_param_dict,
        **_kwargs_CFG
    )
    clusterer.run()


def visualizer_proc(_run_flag, _track_queue, _shared_param_dict, **_kwargs_CFG):
    """
    Simple visualizer that prints tracked objects
    """
    import os
    import time
    
    frame_count = 0
    
    print('[Visualizer] Starting...')
    
    while _run_flag.value:
        try:
            if not _track_queue.empty():
                frame = _track_queue.get(block=False)
                tracks = frame.get('tracks', [])
                
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print("="*80)
                print(f"UNIFIED POINT CLOUD TRACKING - Frame {frame['frame_number']}")
                print(f"Time: {time.strftime('%H:%M:%S')} | Total Points: {frame.get('total_points', 0)}")
                print("="*80)
                print(f"\nActive Tracks: {len(tracks)}")
                print("-"*80)
                
                for track in tracks:
                    radar_info = track.get('radar_point_counts', {})
                    radar_str = ', '.join([f"R{k}:{v}pts" for k, v in radar_info.items()])
                    
                    print(f"  Track {track['tid']:3d} | "
                          f"Pos: ({track['posX']:6.2f}, {track['posY']:6.2f}, {track['posZ']:6.2f}) | "
                          f"Vel: ({track['velX']:5.2f}, {track['velY']:5.2f}, {track['velZ']:5.2f}) | "
                          f"Pts: {track['num_points']:3d} | "
                          f"Radars: {track['num_radars']} [{radar_str}]")
                
                print("-"*80)
                print(f"Updates: {tracks[0]['num_updates'] if tracks else 0} | "
                      f"Confidence: {tracks[0]['confidence']:.2f if tracks else 0}")
                print("="*80)
                
                frame_count += 1
            else:
                sleep(0.01)
                
        except Exception as e:
            print(f'[Visualizer] Error: {e}')
            sleep(0.1)
    
    print('[Visualizer] Stopped')


def monitor_proc_method(_run_flag, _queue_list, _shared_param_dict, **_kwargs_CFG):
    """Monitor process - queue monitoring"""
    sync = SyncMonitor(
        run_flag=_run_flag,
        radar_rd_queue_list=_queue_list,
        shared_param_dict=_shared_param_dict,
        **_kwargs_CFG
    )
    sync.run()


if __name__ == '__main__':
    print("="*80)
    print("UNIFIED POINT CLOUD TRACKING (NO TRACK FUSION)")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Number of Radars: {len(RADAR_CFG_LIST)}")
    print("="*80)
    print("\nSimplified Pipeline:")
    print("  [Radar 1] ──┐")
    print("              ├──> [Unified Clusterer] ──> [Tracks]")
    print("  [Radar 2] ──┘    (Merge + DBSCAN)")
    print("\n  • Point clouds from ALL radars merged first")
    print("  • DBSCAN clustering on combined point cloud")
    print("  • NO track fusion needed!")
    print("="*80)
    
    # Shared variables between processes
    run_flag = Manager().Value('b', True)
    
    shared_param_dict = {
        'mansave_flag'       : Manager().Value('c', None),
        'autosave_flag'      : Manager().Value('b', False),
        'compress_video_file': Manager().Value('c', None),
        'email_image'        : Manager().Value('f', None),
        'proc_status_dict'   : Manager().dict(),
        'save_queue'         : Manager().Queue(maxsize=2000)
    }
    
    # Queues
    pointcloud_queue_list = []  # One queue per radar (radar → clusterer)
    track_output_queue = Manager().Queue()  # Single output queue (clusterer → visualizer)
    proc_list = []
    
    print("\nInitializing pipeline...")
    print("-"*80)
    
    # Create radar reader processes (one per radar)
    for i, RADAR_CFG in enumerate(RADAR_CFG_LIST):
        radar_name = RADAR_CFG['name']
        
        # Queue for this radar's point cloud
        pointcloud_queue = Manager().Queue()
        pointcloud_queue_list.append(pointcloud_queue)
        
        # Config for this radar
        kwargs_CFG_RADAR = {
            'RADAR_CFG': RADAR_CFG,
            'FRAME_EARLY_PROCESSOR_CFG': FRAME_EARLY_PROCESSOR_CFG,
        }
        
        # Radar Reader Process
        radar_proc = Process(
            target=radar_pointcloud_proc,
            args=(run_flag, pointcloud_queue, shared_param_dict),
            kwargs=kwargs_CFG_RADAR,
            name=f"Reader_{radar_name}"
        )
        proc_list.append(radar_proc)
        print(f"  ✓ Radar {i+1}: {radar_name} → {RADAR_CFG['data_port_name']}")
    
    print("-"*80)
    
    # Single Unified Clusterer Process (receives from all radars)
    kwargs_CFG_CLUSTERER = {
        'POINTCLOUD_CLUSTERING_CFG': POINTCLOUD_CLUSTERING_CFG
    }
    
    clusterer_proc = Process(
        target=unified_clusterer_proc,
        args=(run_flag, pointcloud_queue_list, track_output_queue, shared_param_dict),
        kwargs=kwargs_CFG_CLUSTERER,
        name="UnifiedClusterer"
    )
    proc_list.append(clusterer_proc)
    print(f"  ✓ Unified Clusterer (merges {len(RADAR_CFG_LIST)} radars)")
    
    # Visualizer Process
    vis_proc = Process(
        target=visualizer_proc,
        args=(run_flag, track_output_queue, shared_param_dict),
        kwargs={},
        name="Visualizer"
    )
    proc_list.append(vis_proc)
    print(f"  ✓ Visualizer")
    
    # Monitor Process
    kwargs_CFG_MONITOR = {'SYNC_MONITOR_CFG': SYNC_MONITOR_CFG}
    monitor_proc = Process(
        target=monitor_proc_method,
        args=(run_flag, pointcloud_queue_list, shared_param_dict),
        kwargs=kwargs_CFG_MONITOR,
        name="Monitor"
    )
    proc_list.append(monitor_proc)
    print(f"  ✓ Queue Monitor")
    
    # Start all processes
    print("\n" + "="*80)
    print("STARTING PROCESSES...")
    print("="*80)
    for proc in proc_list:
        proc.start()
        print(f"  Started: {proc.name}")
        sleep(0.3)
    
    print("\n" + "="*80)
    print("SYSTEM RUNNING")
    print("="*80)
    print("\nDBSCAN Parameters:")
    print(f"  eps: {POINTCLOUD_CLUSTERING_CFG['eps']}m")
    print(f"  min_samples: {POINTCLOUD_CLUSTERING_CFG['min_samples']}")
    print(f"  min_points_per_cluster: {POINTCLOUD_CLUSTERING_CFG['min_points_per_cluster']}")
    print(f"  Height filter: {POINTCLOUD_CLUSTERING_CFG['min_cluster_height']} - {POINTCLOUD_CLUSTERING_CFG['max_cluster_height']}m")
    print(f"  Track timeout: {POINTCLOUD_CLUSTERING_CFG['track_timeout']}s")
    print("\nPress Ctrl+C to stop...")
    print("="*80)
    
    try:
        while run_flag.value:
            sleep(1)
            
            # Check process health
            alive_count = sum([proc.is_alive() for proc in proc_list])
            if alive_count < len(proc_list):
                print(f"\n⚠️  Warning: {len(proc_list) - alive_count} process(es) died")
                for proc in proc_list:
                    if not proc.is_alive():
                        print(f"  Dead: {proc.name}")
    
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("SHUTTING DOWN...")
        print("="*80)
        run_flag.value = False
        sleep(2)
        
        for proc in proc_list:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                print(f"  Stopped: {proc.name}")
        
        print("\n✓ All processes stopped")
        print("="*80)
