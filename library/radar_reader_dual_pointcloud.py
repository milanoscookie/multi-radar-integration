"""
RadarReader for dual IWR6843AOP with Point Cloud TLVs
Parses TLV 1 (Detected Points) + TLV 7 (Side Info)
Transforms coordinates using FEP, then queues

TLV Types supported:
- Type 1: MMWDEMO_OUTPUT_MSG_DETECTED_POINTS (Cartesian x,y,z,doppler)
- Type 7: MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO (SNR, noise)
"""

import struct
import time
from datetime import datetime
import numpy as np
import serial

from library.frame_early_processor import FrameEProcessor

# TLV Header constants
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
HEADER_LENGTH = 40  # bytes (8 magic + 32 header info)

# TLV Types for Point Cloud
TLV_TYPE_DETECTED_POINTS = 1              # Cartesian detected points (x, y, z, doppler)
TLV_TYPE_DETECTED_POINTS_SIDE_INFO = 7    # Side info (SNR, noise) for each point


class RadarReaderPointCloud:
    """
    Radar reader for point cloud data from IWR6843AOP.
    Parses raw point cloud TLVs and transforms to global coordinates.
    """
    
    def __init__(self, run_flag, radar_rd_queue, shared_param_dict, **kwargs_CFG):
        """
        Initialize radar reader for point cloud parsing + transformation
        
        Args:
            run_flag: multiprocessing.Value for run control
            radar_rd_queue: Queue to put parsed frames
            shared_param_dict: Shared status dictionary
            kwargs_CFG: Configuration dict with RADAR_CFG
        """
        self.run_flag = run_flag
        self.radar_rd_queue = radar_rd_queue
        self.status = shared_param_dict['proc_status_dict']
        
        # Get radar config
        RDR_CFG = kwargs_CFG['RADAR_CFG']
        self.name = RDR_CFG['name']
        self.cfg_port_name = RDR_CFG['cfg_port_name']
        self.data_port_name = RDR_CFG['data_port_name']
        self.cfg_file_name = RDR_CFG['cfg_file_name']
        
        self.status[self.name] = True
        
        # Frame Early Processor for coordinate transformation
        self.fep = FrameEProcessor(**kwargs_CFG)
        
        self.cfg_port = None
        self.data_port = None
        
        # Statistics
        self.frame_count = 0
        self.parse_errors = 0
        self.total_points = 0
        
        self._log('Initialized for Point Cloud TLVs (1, 7) - Parse → Transform → Queue')

    def connect(self) -> bool:
        """Connect to radar COM ports"""
        try:
            self.cfg_port = serial.Serial(self.cfg_port_name, baudrate=115200, timeout=1)
            self.data_port = serial.Serial(self.data_port_name, baudrate=921600, timeout=1)
            
            if not (self.cfg_port.is_open and self.data_port.is_open):
                raise serial.SerialException("Ports not opened")
            
            self._log(f'Connected: CFG={self.cfg_port_name}, DATA={self.data_port_name}')
            # Send configuration
            self._send_cfg(self._read_cfg(self.cfg_file_name), self.cfg_port)
            return True
            
        except serial.SerialException as e:
            self._log(f'Connection failed: {e}')
            return False

    def run(self):
        """Main loop - read, parse point cloud TLVs, transform, queue"""
        if not self.connect():
            self._log(f"Radar {self.name} Connection Failed")
            self.run_flag.value = False
            return

        data_buffer = b''
        self._log('Starting point cloud data acquisition...')
        
        while self.run_flag.value:
            try:
                # Read available data
                bytes_available = self.data_port.in_waiting
                if bytes_available > 0:
                    data_buffer += self.data_port.read(bytes_available)
                
                # Look for complete frame
                if data_buffer.count(MAGIC_WORD) >= 2:
                    # Find first magic word
                    start_idx = data_buffer.find(MAGIC_WORD)
                    if start_idx == -1:
                        continue
                    
                    # Extract frame starting from magic word
                    frame_data = data_buffer[start_idx:]
                    
                    # Parse the frame to get point cloud
                    point_cloud, frame_info, frame_length = self._parse_point_cloud_frame(frame_data)
                    
                    if point_cloud is not None and len(point_cloud['points']) > 0:
                        try:
                            # Apply coordinate transformation
                            transformed_cloud = self._transform_point_cloud(point_cloud)
                            
                            # Put transformed point cloud into queue
                            frame_dict = {
                                'radar_name': self.name,
                                'frame_number': frame_info['frame_number'],
                                'num_detected_obj': frame_info['num_detected_obj'],
                                'num_points': len(transformed_cloud['points']),
                                'point_cloud': transformed_cloud,
                                'timestamp': time.time()
                            }
                            
                            self.radar_rd_queue.put(frame_dict)
                            self.total_points += len(transformed_cloud['points'])
                            
                        except Exception as e:
                            self._log(f'Transform error: {e}')
                        
                        self.frame_count += 1
                        
                        if self.frame_count % 100 == 0:
                            avg_points = self.total_points / max(1, self.frame_count)
                            self._log(f'Frames: {self.frame_count}, Avg points/frame: {avg_points:.1f}, Errors: {self.parse_errors}')
                    
                    # Remove processed frame from buffer
                    if frame_length > 0:
                        data_buffer = data_buffer[start_idx + frame_length:]
                    else:
                        data_buffer = data_buffer[start_idx + len(MAGIC_WORD):]
                
                # Prevent buffer overflow
                if len(data_buffer) > 100000:
                    self._log(f'Warning: Buffer overflow, clearing')
                    data_buffer = b''
                    
            except Exception as e:
                self._log(f'Error in main loop: {e}')
                self.parse_errors += 1
                time.sleep(0.01)

    def _parse_point_cloud_frame(self, data):
        """
        Parse a single frame containing point cloud TLVs
        
        Returns: 
            (point_cloud_dict, frame_info_dict, frame_length) or (None, {}, 0) on error
            
        point_cloud_dict contains:
            - points: list of point dicts with x, y, z, doppler, (snr, noise optional)
            - coordinate_type: 'cartesian' or 'spherical'
        """
        try:
            if len(data) < HEADER_LENGTH:
                return None, {}, 0
            
            # Parse frame header (40 bytes)
            header = data[8:HEADER_LENGTH]
            
            version = struct.unpack('I', header[0:4])[0]
            total_packet_len = struct.unpack('I', header[4:8])[0]
            platform = struct.unpack('I', header[8:12])[0]
            frame_number = struct.unpack('I', header[12:16])[0]
            time_cpu_cycles = struct.unpack('I', header[16:20])[0]
            num_detected_obj = struct.unpack('I', header[20:24])[0]
            num_tlvs = struct.unpack('I', header[24:28])[0]
            subframe_number = struct.unpack('I', header[28:32])[0]
            
            frame_info = {
                'version': version,
                'total_packet_len': total_packet_len,
                'platform': platform,
                'frame_number': frame_number,
                'time_cpu_cycles': time_cpu_cycles,
                'num_detected_obj': num_detected_obj,
                'num_tlvs': num_tlvs,
                'subframe_number': subframe_number
            }
            
            if len(data) < total_packet_len:
                return None, {}, 0
            
            # Parse TLVs
            tlv_start = HEADER_LENGTH
            points_cartesian = None
            side_info = None
            
            for _ in range(num_tlvs):
                if tlv_start + 8 > len(data):
                    break
                
                tlv_type = struct.unpack('I', data[tlv_start:tlv_start+4])[0]
                tlv_length = struct.unpack('I', data[tlv_start+4:tlv_start+8])[0]
                tlv_data_start = tlv_start + 8
                tlv_data = data[tlv_data_start:tlv_data_start+tlv_length]
                
                # Parse based on TLV type
                if tlv_type == TLV_TYPE_DETECTED_POINTS:
                    points_cartesian = self._parse_tlv_detected_points(tlv_data)
                    
                elif tlv_type == TLV_TYPE_DETECTED_POINTS_SIDE_INFO:
                    side_info = self._parse_tlv_side_info(tlv_data)
                
                tlv_start = tlv_data_start + tlv_length
            
            # Build point cloud result
            point_cloud = {'points': [], 'coordinate_type': 'cartesian'}
            
            if points_cartesian is not None:
                point_cloud['points'] = points_cartesian
                
                # Merge side info if available
                if side_info is not None and len(side_info) == len(points_cartesian):
                    for i, pt in enumerate(point_cloud['points']):
                        pt['snr'] = side_info[i]['snr']
                        pt['noise'] = side_info[i]['noise']
            
            return point_cloud, frame_info, total_packet_len
            
        except Exception as e:
            self._log(f'Parse error: {e}')
            self.parse_errors += 1
            return None, {}, 0

    def _parse_tlv_detected_points(self, tlv_data):
        """
        Parse TLV Type 1: Detected Points (Cartesian)
        Each point: 16 bytes (x, y, z, doppler as floats)
        
        Returns: list of point dicts with x, y, z, doppler
        """
        POINT_SIZE = 16
        num_points = len(tlv_data) // POINT_SIZE
        
        points = []
        for i in range(num_points):
            offset = i * POINT_SIZE
            point_data = tlv_data[offset:offset + POINT_SIZE]
            
            if len(point_data) < POINT_SIZE:
                break
            
            try:
                x = struct.unpack('f', point_data[0:4])[0]
                y = struct.unpack('f', point_data[4:8])[0]
                z = struct.unpack('f', point_data[8:12])[0]
                doppler = struct.unpack('f', point_data[12:16])[0]
                
                point = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'doppler': doppler,
                    'snr': None,
                    'noise': None
                }
                points.append(point)
                
            except Exception as e:
                self._log(f'Error parsing point {i}: {e}')
                continue
        
        return points

    def _parse_tlv_side_info(self, tlv_data):
        """
        Parse TLV Type 7: Side Info for Detected Points
        Each point: 4 bytes (snr, noise as uint16)
        Values are in 0.1 dB units
        
        Returns: list of dicts with snr, noise
        """
        SIDE_INFO_SIZE = 4
        num_points = len(tlv_data) // SIDE_INFO_SIZE
        
        side_info = []
        for i in range(num_points):
            offset = i * SIDE_INFO_SIZE
            info_data = tlv_data[offset:offset + SIDE_INFO_SIZE]
            
            if len(info_data) < SIDE_INFO_SIZE:
                break
            
            try:
                snr_raw = struct.unpack('H', info_data[0:2])[0]
                noise_raw = struct.unpack('H', info_data[2:4])[0]
                
                # Convert from 0.1 dB to dB
                snr = snr_raw * 0.1
                noise = noise_raw * 0.1
                
                side_info.append({
                    'snr': snr,
                    'noise': noise
                })
                
            except Exception as e:
                self._log(f'Error parsing side info {i}: {e}')
                continue
        
        return side_info

    def _transform_point_cloud(self, point_cloud):
        """
        Apply coordinate transformation using FEP (rotation + translation)
        
        Args:
            point_cloud: dict with 'points' list and 'coordinate_type'
        
        Returns:
            Transformed point cloud dict
        """
        points = point_cloud['points']
        
        if len(points) == 0:
            return point_cloud
        
        # Extract positions as numpy array
        positions = np.array([[pt['x'], pt['y'], pt['z']] for pt in points], dtype=np.float32)
        
        # Apply rotation
        pos_rotated = self.fep.FEP_trans_rotation_3D(positions)
        
        # Apply translation
        pos_transformed = self.fep.FEP_trans_position_3D(pos_rotated)
        
        # Build transformed point cloud
        transformed_points = []
        for i, pt in enumerate(points):
            transformed_pt = {
                'x': pos_transformed[i, 0],
                'y': pos_transformed[i, 1],
                'z': pos_transformed[i, 2],
                'doppler': pt['doppler'],
                'snr': pt.get('snr'),
                'noise': pt.get('noise'),
                'radar_name': self.name
            }
            transformed_points.append(transformed_pt)
        
        return {
            'points': transformed_points,
            'coordinate_type': point_cloud['coordinate_type']
        }

    def _read_cfg(self, cfg_file_name):
        """Read radar configuration file"""
        cfg_list = []
        try:
            with open(cfg_file_name) as f:
                lines = f.read().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('%'):
                    cfg_list.append(line)
        except Exception as e:
            self._log(f'Error reading config: {e}')
        return cfg_list

    def _send_cfg(self, cfg_list, cfg_port):
        """Send configuration commands to radar"""
        for line in cfg_list:
            try:
                cfg_port.write((line + '\n').encode())
                time.sleep(0.05)
                
                # Read response
                response = b''
                timeout = time.time() + 1
                while time.time() < timeout:
                    if cfg_port.in_waiting > 0:
                        response += cfg_port.read(cfg_port.in_waiting)
                        break
                    time.sleep(0.01)
                
                self._log(f'CFG: {line[:50]}... -> {response.decode("utf-8", errors="ignore").strip()}')
                
            except Exception as e:
                self._log(f'Config send error: {e}')

    def _log(self, txt):
        """Print with radar name prefix"""
        print(f'[{self.name}]\t{txt}')

    def __del__(self):
        """Cleanup on exit"""
        try:
            if self.cfg_port and self.cfg_port.is_open:
                self.cfg_port.write(b'sensorStop\n')
                time.sleep(0.1)
                self.cfg_port.close()
            if self.data_port and self.data_port.is_open:
                self.data_port.close()
        except:
            pass
        
        avg_points = self.total_points / max(1, self.frame_count)
        self._log(f"Closed. Frames: {self.frame_count}, Avg points: {avg_points:.1f}, Errors: {self.parse_errors}")
        self._log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        
        if self.name in self.status:
            self.status[self.name] = False
