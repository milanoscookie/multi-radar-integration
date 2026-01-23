"""
Point Cloud Radar with DBSCAN Clustering Pipeline

This script demonstrates the separated pipeline:
1. RadarReaderPointCloud - Reads point cloud TLVs (1, 7) from radar
2. PointCloudClustererProcess - DBSCAN clustering to extract objects
3. FuseDualRadar (TrackFusion) - Fuses tracks across radars

Pipeline:
[Radar HW] → [RadarReader] → [Clusterer] → [TrackFusion] → [Visualizer]
              Point Cloud     DBSCAN        EKF Fusion      Output
"""

from multiprocessing import Process, Manager
from time import sleep

# Import the radar reader (point cloud only - no clustering)
from library.radar_reader_dual_pointcloud import RadarReaderPointCloud

# Import the separate clusterer process
from library.pointcloud_clusterer_process import PointCloudClustererProcess

# Import fusion and visualization
from library.sync_monitor import SyncMonitor
from library.visualizer_dual_tracks import FuseDualRadar

# Import configuration
from cfg.config_demo_dual_radar import *


def radar_pointcloud_proc(_run_flag, _radar_rd_queue, _shared_param_dict, **_kwargs_CFG):
    """
    Process 1: Radar Reader
    Reads raw point cloud TLVs and outputs point cloud frames
    """
    radar = RadarReaderPointCloud(
        run_flag=_run_flag,
        radar_rd_queue=_radar_rd_queue,
        shared_param_dict=_shared_param_dict,
        **_kwargs_CFG
    )
    radar.run()


def clusterer_proc(_run_flag, _input_queue, _output_queue, _shared_param_dict, **_kwargs_CFG):
    """
    Process 2: DBSCAN Clusterer
    Takes point cloud, outputs tracked objects
    """
    clusterer = PointCloudClustererProcess(
        run_flag=_run_flag,
        input_queue=_input_queue,
        output_queue=_output_queue,
        shared_param_dict=_shared_param_dict,
        **_kwargs_CFG
    )
    clusterer.run()


def fuse_vis_dualradar(_run_flag, _radar_rd_queue_list, vis_queue, _shared_param_dict, **_kwargs_CFG):
    """
    Process 3: Track Fusion & Visualization
    Fuses tracked objects from multiple radars
    """
    fuser = FuseDualRadar(
        run_flag=_run_flag,
        radar_rd_queue_list=_radar_rd_queue_list,
        vis_queue=vis_queue,
        shared_param_dict=_shared_param_dict,
        **_kwargs_CFG
    )
    fuser.run()


def monitor_proc_method(_run_flag, _radar_rd_queue_list, _shared_param_dict, **_kwargs_CFG):
    """Monitor process - queue sync and status"""
    sync = SyncMonitor(
        run_flag=_run_flag,
        radar_rd_queue_list=_radar_rd_queue_list,
        shared_param_dict=_shared_param_dict,
        **_kwargs_CFG
    )
    sync.run()


if __name__ == '__main__':
    print("="*80)
    print("DUAL RADAR POINT CLOUD TRACKING - SEPARATED PIPELINE")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Number of Radars: {len(RADAR_CFG_LIST)}")
    print("="*80)
    print("\nPipeline (3 stages per radar):")
    print("  Stage 1: [RadarReader]  → Parse TLV 1,7 → Transform (FEP) → Point Cloud")
    print("  Stage 2: [Clusterer]    → DBSCAN → Tracked Objects")
    print("  Stage 3: [TrackFusion]  → Cross-Radar EKF Fusion → Output")
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
    
    # Lists for queues and processes
    pointcloud_queue_list = []   # Radar → Clusterer (point clouds)
    tracked_obj_queue_list = []  # Clusterer → Fusion (tracked objects)
    proc_list = []
    
    print("\nInitializing 3-stage pipeline for each radar...")
    print("-"*80)
    
    for i, RADAR_CFG in enumerate(RADAR_CFG_LIST):
        radar_name = RADAR_CFG['name']
        print(f"\n  Radar {i+1}: {radar_name}")
        
        # Queue 1: Radar Reader → Clusterer (point clouds)
        pointcloud_queue = Manager().Queue()
        pointcloud_queue_list.append(pointcloud_queue)
        
        # Queue 2: Clusterer → Track Fusion (tracked objects)
        tracked_obj_queue = Manager().Queue()
        tracked_obj_queue_list.append(tracked_obj_queue)
        
        # Config for this radar
        kwargs_CFG_RADAR = {
            'RADAR_CFG': RADAR_CFG,
            'FRAME_EARLY_PROCESSOR_CFG': FRAME_EARLY_PROCESSOR_CFG,
        }
        
        kwargs_CFG_CLUSTERER = {
            'RADAR_CFG': RADAR_CFG,
            'POINTCLOUD_CLUSTERING_CFG': POINTCLOUD_CLUSTERING_CFG
        }
        
        # Process 1: Radar Reader (outputs point cloud)
        radar_proc = Process(
            target=radar_pointcloud_proc,
            args=(run_flag, pointcloud_queue, shared_param_dict),
            kwargs=kwargs_CFG_RADAR,
            name=f"Reader_{radar_name}"
        )
        proc_list.append(radar_proc)
        print(f"    ✓ Stage 1: RadarReader → {RADAR_CFG['data_port_name']}")
        
        # Process 2: Clusterer (input: point cloud, output: tracked objects)
        clusterer_proc_inst = Process(
            target=clusterer_proc,
            args=(run_flag, pointcloud_queue, tracked_obj_queue, shared_param_dict),
            kwargs=kwargs_CFG_CLUSTERER,
            name=f"Clusterer_{radar_name}"
        )
        proc_list.append(clusterer_proc_inst)
        print(f"    ✓ Stage 2: DBSCAN Clusterer")
    
    print("-"*80)
    
    # Process 3: Track Fusion (receives from all clusterers)
    vis_queue = Manager().Queue()
    kwargs_CFG_FUSE = {
        'RADAR_CFG_LIST': RADAR_CFG_LIST,
        'TRACK_FUSION_CFG': TRACK_FUSION_CFG,
        'VISUALIZER_CFG': VISUALIZER_CFG,
        'INDUSTRIAL_VIS_CFG': INDUSTRIAL_VIS_CFG
    }
    
    fuser_proc = Process(
        target=fuse_vis_dualradar,
        args=(run_flag, tracked_obj_queue_list, vis_queue, shared_param_dict),
        kwargs=kwargs_CFG_FUSE,
        name="TrackFusion"
    )
    proc_list.append(fuser_proc)
    print(f"\n  ✓ Stage 3: TrackFusion (EKF) - receives from all clusterers")
    
    # Monitor process (monitors clusterer output queues)
    kwargs_CFG_MONITOR = {'SYNC_MONITOR_CFG': SYNC_MONITOR_CFG}
    monitor_proc = Process(
        target=monitor_proc_method,
        args=(run_flag, tracked_obj_queue_list, shared_param_dict),
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
    print("\nProcess Architecture:")
    for i, RADAR_CFG in enumerate(RADAR_CFG_LIST):
        print(f"  {RADAR_CFG['name']}:")
        print(f"    [RadarReader] --pointcloud--> [Clusterer] --tracks--> [TrackFusion]")
    print("\nDBSCAN Parameters:")
    print(f"  eps: {POINTCLOUD_CLUSTERING_CFG['eps']}m, min_samples: {POINTCLOUD_CLUSTERING_CFG['min_samples']}")
    print("\nTrack Fusion Parameters:")
    print(f"  Distance: {TRACK_FUSION_CFG['distance_threshold']}m, Velocity: {TRACK_FUSION_CFG['velocity_threshold']}m/s")
    print("\nPress Ctrl+C to stop...")
    print("="*80)
    
    try:
        # Main loop - monitor processes
        while run_flag.value:
            sleep(1)
            
            # Check if any process has died
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
        
        # Terminate all processes
        for proc in proc_list:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                print(f"  Stopped: {proc.name}")
        
        print("\n✓ All processes stopped")
        print("="*80)
