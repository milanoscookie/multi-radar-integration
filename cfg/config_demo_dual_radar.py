"""
Config file for Dual IWR6843AOP 3D People Tracking
Modified for 2 radars with TLV 1010 output only
"""

# global save parameters
EXPERIMENT_NAME = 'dual_radar_tracking'
RADAR_FPS = 20  # 20 frames per second for People Tracking demo

# Save settings (can disable if only live viewing)
MANSAVE_ENABLE = False
MANSAVE_PERIOD = 30

AUTOSAVE_ENABLE = False
AUTOSAVE_PERIOD = 600

# =============================================================================
# DUAL RADAR CONFIGURATION
# =============================================================================
RADAR_CFG_LIST = [
    {
        'name'          : 'IWR6843_Radar1',
        'cfg_port_name' : '/dev/ttyUSB0',
        'data_port_name': '/dev/ttyUSB1',
        'cfg_file_name' : './cfg/AOP_6m_default.cfg',
        
        # Coordinate transformation for Radar 1
        'pos_offset'    : (0, 0, 2.5),      # (x, y, z) position in meters
        'facing_angle'  : {
            'angle': (-20, 0, 0),           # (pitch, yaw, roll) in degrees
            'sequence': 'zyx'               # Rotation order
        },
        
        # Boundary limits (from radar's local view, before transformation)
        'xlim'          : (-2, 2),         # Left-right limit
        'ylim'          : (0.2, 4.1),      # Depth limit (front of radar)
        'zlim'          : (0, 3.7),        # Height limit
        
        # Energy threshold (not used for 1010, but keep for compatibility)
        'ES_threshold'  : {'range': (0, None), 'speed_none_0_exception': False},
    },
    
    # Radar 2 - Right side (example placement)
    # {
    #     'name'          : 'IWR6843_Radar2',
    #     'cfg_port_name' : 'COM5',        # ⚠️ UPDATE: Your second radar's config port
    #     'data_port_name': 'COM6',        # ⚠️ UPDATE: Your second radar's data port
    #     'cfg_file_name' : './cfg/IWR6843_3D_20fps_people_tracking.cfg',
        
    #     # Coordinate transformation for Radar 2
    #     'pos_offset'    : (4, 0, 2.5),   # ⚠️ UPDATE: (x, y, z) position in meters
    #     'facing_angle'  : {
    #         'angle': (0, -45, 0),        # ⚠️ UPDATE: Example: facing -45° (towards left)
    #         'sequence': 'zyx'
    #     },
        
    #     # Boundary limits (from radar's local view, before transformation)
    #     'xlim'          : (-3, 3),
    #     'ylim'          : (0.5, 8),
    #     'zlim'          : (0, 3),
        
    #     'ES_threshold'  : {'range': (0, None), 'speed_none_0_exception': False},
    # },
]

# =============================================================================
# SYNC MONITOR CONFIG
# =============================================================================
SYNC_MONITOR_CFG = {
    'rd_qsize_warning': 5,   # Warning when queue size exceeds this
    'sc_qsize_warning': 20,
}

# =============================================================================
# FRAME EARLY PROCESSOR CONFIG (Per Radar)
# =============================================================================
FRAME_EARLY_PROCESSOR_CFG = {
    'FEP_frame_deque_length': 1,  # For 1010, we don't accumulate frames (tracks are already processed)
}

# =============================================================================
# VISUALIZER CONFIG (For Industrial Visualizer output)
# =============================================================================
VISUALIZER_CFG = {
    'dimension'               : '3D',
    
    # Global coordinate limits (after merging both radars)
    'VIS_xlim'                : (-2, 2),    # ⚠️ UPDATE: Based on your room size
    'VIS_ylim'                : (0.3, 4),
    'VIS_zlim'                : (0, 3.7),
    
    'auto_inactive_skip_frame': 0,  # No skipping for tracking data
    
    # Output settings
    'output_format'           : 'industrial_visualizer',  # or 'tlv', 'json'
    'output_port'             : 5000,  # Port for Industrial Visualizer
}

# =============================================================================
# FRAME POST PROCESSOR CONFIG (Global fusion)
# =============================================================================
FRAME_POST_PROCESSOR_CFG = {
    # Global coordinate limits (merged view)
    'FPP_global_xlim' : (-5, 5),
    'FPP_global_ylim' : (0, 10),
    'FPP_global_zlim' : (0.1, 3),
    
    # Track ID offset for radar 2 to avoid conflicts with radar 1 local IDs
    'track_id_offset' : 1000,
}

# =============================================================================
# TRACK FUSION CONFIG (EKF-CA Model with Track Identity Preservation)
# =============================================================================
TRACK_FUSION_CFG = {
    'enable'                  : True,
    
    # Association thresholds (BOTH must be satisfied for NEW tracks to merge)
    'distance_threshold'      : 0.5,   # meters - max position distance for initial fusion
    'velocity_threshold'      : 1.0,   # m/s - max velocity difference for initial fusion
    
    # TRACK IDENTITY PRESERVATION (prevents merging when people get close)
    # Once a local track is associated with a global track, that association is LOCKED
    # until the track times out. New fusion only happens in "clear" areas.
    'min_track_separation'    : 0.3,   # meters - if existing tracks are closer than 3x this,
                                        #          new measurements won't be cross-radar fused
    
    # TRACK SPLITTING (recovers from incorrect merges)
    # If measurements from different radars on the SAME global track diverge,
    # the track will be split back into separate tracks
    'split_distance_threshold': 0.8,   # meters - split if positions diverge more than this
    'split_velocity_threshold': 0.8,   # m/s - split if velocities diverge more than this
    
    # EKF process noise (Constant Acceleration model)
    'sigma_j'                 : 1.0,   # m/s^3 - jerk std dev (increase for maneuvering targets)
    
    # Measurement noise
    'meas_sigma_pos'          : 0.25,  # meters - position measurement noise
    'meas_sigma_vel'          : 0.50,  # m/s - velocity measurement noise
    'meas_sigma_acc'          : 1.0,   # m/s^2 - acceleration measurement noise
    
    # Outlier rejection
    'use_mahalanobis_gating'  : True,
    'mahalanobis_gate'        : 22.0,  # chi2(9)@99% for full state gating
    
    # Track management
    'tid_timeout'             : 2.0,   # seconds - drop track after no detections
}

# =============================================================================
# INDUSTRIAL VISUALIZER OUTPUT CONFIG
# =============================================================================
INDUSTRIAL_VIS_CFG = {
    'enable'          : True,
    'output_protocol' : 'udp',     # 'udp' or 'tcp'
    'ip_address'      : '127.0.0.1',
    'port'            : 5000,
    'frame_format'    : 'tlv',     # Keep TLV format for Industrial Visualizer
    'include_tracks'  : True,
    'include_points'  : False,     # We only have tracks (1010), no point cloud
}

# =============================================================================
# UNUSED CONFIGS (Keep for compatibility)
# =============================================================================
DBSCAN_GENERATOR_CFG = {'Default': {}}  # Not used for 1010
BGNOISE_FILTER_CFG = {'BGN_enable': False}
HUMAN_TRACKING_CFG = {'TRK_enable': False}  # Tracking done on-chip
HUMAN_OBJECT_CFG = {}
SAVE_CENTER_CFG = {
    'file_save_dir'      : f'./data/{EXPERIMENT_NAME}/',
    'experiment_name'    : EXPERIMENT_NAME,
    'mansave_period'     : MANSAVE_PERIOD,
    'mansave_rdr_frame_max': int(MANSAVE_PERIOD * RADAR_FPS * 1.2),
}
CAMERA_CFG = {}
EMAIL_NOTIFIER_CFG = {}