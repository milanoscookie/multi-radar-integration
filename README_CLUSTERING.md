# Point Cloud DBSCAN Clustering for Multi-Radar Tracking

## Overview

This module converts raw point cloud data (TLV 1, TLV 7) from mmWave radar into tracked objects using DBSCAN clustering. The modules are **separated** for clean architecture:

1. **RadarReaderPointCloud** - Parses TLVs, outputs point clouds
2. **PointCloudClustererProcess** - DBSCAN clustering, outputs tracked objects
3. **TrackFusion** - Cross-radar EKF fusion

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ SEPARATED 3-STAGE PIPELINE (per radar)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Radar Hardware]                                                   │
│         │ Serial UART (TLV 1 + TLV 7)                              │
│         ↓                                                           │
│  ┌─────────────────────────────────────┐                           │
│  │ Stage 1: RadarReaderPointCloud      │  (Process 1)              │
│  │  • Parse Magic Word + Frame Header  │                           │
│  │  • Extract x,y,z,doppler,SNR,noise  │                           │
│  │  • Apply FEP coordinate transform   │                           │
│  └─────────────────┬───────────────────┘                           │
│                    │ pointcloud_queue (Point Cloud Frames)         │
│                    ↓                                                │
│  ┌─────────────────────────────────────┐                           │
│  │ Stage 2: PointCloudClustererProcess │  (Process 2)              │
│  │  • DBSCAN clustering                │                           │
│  │  • Physical filtering (height, vol) │                           │
│  │  • Temporal track association       │                           │
│  └─────────────────┬───────────────────┘                           │
│                    │ tracked_obj_queue (Tracked Objects)           │
│                    ↓                                                │
│  ┌─────────────────────────────────────┐                           │
│  │ Stage 3: TrackFusion (EKF)          │  (Process 3 - shared)     │
│  │  • Cross-radar association          │                           │
│  │  • Kalman filter predict/update     │                           │
│  │  • Late fusion, track splitting     │                           │
│  └─────────────────┬───────────────────┘                           │
│                    ↓                                                │
│  [Fused Global Tracks → Visualizer/Output]                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Files

### Core Modules (Separated)

| File | Stage | Responsibility |
|------|-------|----------------|
| `library/radar_reader_dual_pointcloud.py` | 1 | Parse TLV 1,7 → Point Cloud |
| `library/pointcloud_clusterer_process.py` | 2 | DBSCAN → Tracked Objects |
| `library/track_fusion_pkg/` | 3 | EKF Cross-Radar Fusion |

## Output Format

Clustered objects output from `PointCloudClusterer` matches the track fusion input format:

```python
{
    'tid': 5,                    # Track ID (assigned by clusterer)
    'posX': 1.25,                # Centroid X position (meters)
    'posY': 2.30,                # Centroid Y position (meters)
    'posZ': 1.50,                # Centroid Z position (meters)
    'velX': 0.48,                # Velocity X (m/s, from doppler)
    'velY': 0.30,                # Velocity Y (m/s, from doppler)
    'velZ': 0.0,                 # Velocity Z (m/s)
    'accX': 0.0,                 # Acceleration (not estimated)
    'accY': 0.0,
    'accZ': 0.0,
    'confidence': 0.75,          # Based on SNR
    'num_points': 42,            # Points in cluster
    'avg_snr': 15.2,             # Average SNR (dB)
    'dimensions': [0.6, 0.4, 1.7]  # [width, depth, height] in meters
}
```

## Algorithm Details

### DBSCAN Clustering

**Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** groups points that are closely packed together:

1. **Core Points**: Points with at least `min_samples` neighbors within `eps` radius
2. **Border Points**: Points within `eps` of a core point
3. **Noise Points**: Points that are neither core nor border (rejected)

**Advantages**:
- No need to specify number of clusters (unlike k-means)
- Can find arbitrarily shaped clusters
- Robust to noise and outliers
- Fast: O(n log n) with spatial indexing

### Temporal Tracking

Simple centroid-based tracking across frames:

1. **Association**: Match new clusters to existing tracks using nearest centroid
2. **Update**: Exponential moving average for smoothing
3. **Creation**: New tracks for unassociated clusters
4. **Timeout**: Remove tracks not updated within timeout threshold

### Velocity Estimation

Doppler velocity is radial (towards/away from radar). To convert to 3D velocity:

```
velocity_vector = normalized(centroid_position) * avg_doppler
```

This is a rough approximation. For better accuracy, use multiple radars or track velocity via finite differences.

## Tuning Guide

### For Dense Crowds (Multiple People Close Together)

```python
POINTCLOUD_CLUSTERING_CFG = {
    'eps': 0.3,              # Smaller radius → tighter clusters
    'min_samples': 5,        # More points → fewer false positives
    'min_points_per_cluster': 10,
}
```

### For Sparse Environments (Few People, Far Apart)

```python
POINTCLOUD_CLUSTERING_CFG = {
    'eps': 0.7,              # Larger radius → captures spread-out points
    'min_samples': 3,        # Fewer points → more sensitive
    'min_points_per_cluster': 3,
}
```

### For Fast-Moving Targets

```python
POINTCLOUD_CLUSTERING_CFG = {
    'max_track_distance': 2.5,  # Allow larger movement between frames
    'track_timeout': 0.5,        # Shorter timeout → faster track drops
}
```

## Testing

Run the test suite to validate clustering:

```bash
python test_pointcloud_clustering.py
```

**Test Coverage**:
- ✓ Single person detection
- ✓ Multiple people separation
- ✓ Temporal tracking (same TID across frames)
- ✓ Track timeout
- ✓ Noise rejection

Expected output:
```
TEST 1: Single Person Cluster
  Input points: 50
  Detected objects: 1
  ✓ PASS: Position accurate

TEST 2: Two People (Separated)
  Input points: 80
  Detected objects: 2
  ✓ PASS: Detected 2 objects

TEST 3: Temporal Tracking (Moving Person)
  ✓ PASS: Same track ID maintained across frames

TEST 4: Track Timeout
  ✓ PASS: Track timed out correctly

TEST 5: Noise Point Rejection
  ✓ PASS: Noise rejected, only person detected
```

## Integration with Existing System

The clustered objects are **fully compatible** with the existing track fusion pipeline:

```python
# From main_pointcloud_clustering.py

# Radar process outputs clustered tracks
radar_frame = {
    'radar_name': 'Radar1',
    'tracks': tracked_objects,  # Output from clusterer
    'timestamp': time.time()
}

# Track fusion receives identical format as TLV 1010 reader
fuser = FuseDualRadar(...)
fuser.run()  # No changes needed!
```

The `TrackFusion` module will:
1. Associate tracks across radars
2. Apply EKF filtering
3. Perform late fusion for missed associations
4. Split incorrectly merged tracks

## Comparison: Point Cloud vs. On-Chip Tracking

| Aspect | Point Cloud (TLV 1,7) + DBSCAN | On-Chip Tracking (TLV 1010) |
|--------|-------------------------------|----------------------------|
| **Latency** | Higher (clustering in software) | Lower (tracking on chip) |
| **Flexibility** | Full control over clustering | Fixed tracker parameters |
| **Raw Data** | Access to all points | Only tracks, no points |
| **Customization** | Easy to tune DBSCAN | Requires firmware changes |
| **CPU Load** | Higher (DBSCAN per frame) | Lower (tracker offloaded) |
| **Use Case** | Research, custom algorithms | Production, real-time |

## Troubleshooting

### Too Many Small Clusters

**Problem**: Every small group of points becomes a track

**Solution**: Increase `min_points_per_cluster` or `min_samples`

```python
'min_samples': 5,            # Require more points
'min_points_per_cluster': 8,
```

### Missing Detections

**Problem**: People not detected or tracks drop frequently

**Solution**: Decrease `eps` or `min_samples`, increase `track_timeout`

```python
'eps': 0.7,                  # Larger cluster radius
'min_samples': 3,            # Less strict
'track_timeout': 2.0,        # Longer timeout
```

### Clusters Merge Incorrectly

**Problem**: Two close people form single cluster

**Solution**: Decrease `eps`, add stricter height/volume filters

```python
'eps': 0.4,                  # Tighter clustering
'max_cluster_volume': 2.0,   # Reject large merged clusters
```

### Track IDs Change Frequently

**Problem**: Same person gets new TID every frame

**Solution**: Increase `max_track_distance`

```python
'max_track_distance': 2.0,   # Allow more movement
```

## Performance

Typical performance on modern CPU (i7/Ryzen):

| Metric | Value |
|--------|-------|
| Frame rate | 20 Hz |
| Points per frame | 50-200 |
| DBSCAN time | 1-5 ms |
| Clustering latency | < 10 ms |
| CPU usage | 5-10% per radar |

## Future Enhancements

Possible improvements:

1. **Kalman Filtering in Clusterer**: Track velocity via EKF within clusterer
2. **Multi-Frame Clustering**: Use temporal information in DBSCAN
3. **Better Velocity Estimation**: Use Doppler + multi-radar triangulation
4. **Adaptive DBSCAN**: Adjust `eps` based on local point density
5. **Height-Aware Clustering**: Use different `eps` for horizontal vs vertical
6. **GPU Acceleration**: Offload DBSCAN to GPU for higher throughput

## References

- DBSCAN: Ester et al., "A Density-Based Algorithm for Discovering Clusters", KDD 1996
- Texas Instruments mmWave SDK: TLV Format Documentation
- scikit-learn DBSCAN: https://scikit-learn.org/stable/modules/clustering.html#dbscan
