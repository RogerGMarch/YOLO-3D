#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path

# Set display environment variable for headless environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import our modules
#from detection_model import ObjectDetector
from pose_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from load_camera_params import load_camera_params, apply_camera_params_to_estimator
from scipy.spatial import ConvexHull

import json
from datetime import datetime
import json
from datetime import datetime
import pandas as pd
import os
import json
from datetime import datetime
import pandas as pd
import os

class ParquetExporter:
    """Handles exporting tracking data directly to Parquet format for Houdini import"""
    
    def __init__(self, parquet_output="tracking_data.parquet", buffer_size=2000,flip_y=False):
        """
        Initialize the exporter
        
        Args:
            parquet_output: Path to the output Parquet file
            buffer_size: Number of records to buffer before writing to disk
        """
        self.parquet_output = parquet_output
        self.buffer_size = buffer_size
        self.keypoint_data = []  # Store keypoint data for Parquet export
        self.frame_count = 0
        self.flip_y = flip_y
        
        # Prepare for partial writes if needed
        self.part_count = 0
        self.temp_files = []
        
        # For tracking metrics between frames
        self.prev_frame_data = {}  # Store previous frame data for velocity calculations
        self.prev_group_metrics = {}  # Store previous frame group metrics
        self.smoothing_window = 3  # Number of frames to smooth metrics over
        
        print(f"Parquet exporter initialized with buffer size {buffer_size}")
        print(f"Output will be saved to {parquet_output}")
        print(f"Enhanced choreography metrics will be calculated")

    def calculate_group_metrics(self, boxes_3d):
        """
        Calculate group-level metrics for choreography analysis
        
        Args:
            boxes_3d: List of 3D box dictionaries containing detection data
            
        Returns:
            Dictionary with calculated group metrics
        """
        # Filter for people only
        people = [box for box in boxes_3d if 'person' in box['class_name'].lower()]
        
        if len(people) < 2:
            # Not enough people for group metrics
            return {}
        
        # Calculate center of mass for the group
        com_x, com_y, com_z = 0, 0, 0
        total_keypoints = 0
        
        # Get all valid keypoints across all people
        all_keypoints = []
        
        for person in people:
            if 'keypoints' not in person or person['keypoints'] is None:
                continue
                
            keypoints = person['keypoints']
            keypoint_depths = person.get('keypoint_depths', [])
            
            for i, kpt in enumerate(keypoints):
                x, y, conf = kpt
                depth = keypoint_depths[i] if i < len(keypoint_depths) else None
                
                if conf > 0.5 and depth is not None:
                    all_keypoints.append((x, y, depth, person.get('object_id', -1), i))
                    com_x += x
                    com_y += y
                    com_z += depth
                    total_keypoints += 1
        
        if total_keypoints == 0:
            return {}
        
        # Compute center of mass
        com_x /= total_keypoints
        com_y /= total_keypoints
        com_z /= total_keypoints
        
        # Calculate spatial spread/dispersion (standard deviation from center)
        dist_from_center = []
        for x, y, depth, _, _ in all_keypoints:
            dist = ((x - com_x)**2 + (y - com_y)**2 + (depth - com_z)**2)**0.5
            dist_from_center.append(dist)
        
        spatial_spread = np.std(dist_from_center) if dist_from_center else 0
        
        # Calculate pairwise distances between performers (centroid to centroid)
        person_centroids = []
        for person in people:
            if 'keypoints' not in person or person['keypoints'] is None:
                continue
            
            # Use hip center as person centroid if available
            keypoints = person['keypoints']
            if len(keypoints) >= 17:  # COCO format has 17 keypoints
                # Calculate hip center (midpoint between left hip and right hip)
                left_hip = keypoints[11]  # Index might vary based on your keypoint format
                right_hip = keypoints[12]
                
                if left_hip[2] > 0.5 and right_hip[2] > 0.5:
                    hip_center_x = (left_hip[0] + right_hip[0]) / 2
                    hip_center_y = (left_hip[1] + right_hip[1]) / 2
                    
                    # Get depth at this point
                    hip_center_depth = None
                    if 'keypoint_depths' in person and len(person['keypoint_depths']) > 11:
                        left_depth = person['keypoint_depths'][11]
                        right_depth = person['keypoint_depths'][12]
                        if left_depth is not None and right_depth is not None:
                            hip_center_depth = (left_depth + right_depth) / 2
                    
                    if hip_center_depth is not None:
                        person_centroids.append((hip_center_x, hip_center_y, hip_center_depth, person.get('object_id', -1)))
            
            # Fallback to bbox center if keypoints don't work
            if not person_centroids or person.get('object_id', -1) not in [p[3] for p in person_centroids]:
                x1, y1, x2, y2 = person['bbox_2d']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                depth = person['depth_value']
                person_centroids.append((center_x, center_y, depth, person.get('object_id', -1)))
        
        # Calculate pairwise distances
        pairwise_distances = []
        for i in range(len(person_centroids)):
            for j in range(i+1, len(person_centroids)):
                x1, y1, z1, id1 = person_centroids[i]
                x2, y2, z2, id2 = person_centroids[j]
                distance = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
                pairwise_distances.append((distance, id1, id2))
        
        # Calculate formation metrics
        avg_dist = np.mean([d for d, _, _ in pairwise_distances]) if pairwise_distances else 0
        max_dist = np.max([d for d, _, _ in pairwise_distances]) if pairwise_distances else 0
        min_dist = np.min([d for d, _, _ in pairwise_distances]) if pairwise_distances else 0
        
        # Calculate convex hull area for group formation
        if len(person_centroids) >= 3:
            try:
                points = np.array([(x, y) for x, y, _, _ in person_centroids])
                hull = ConvexHull(points)
                hull_area = hull.volume  # This is actually the area in 2D
            except:
                hull_area = 0
        else:
            hull_area = 0
        
        # Calculate symmetry measure (based on reflection across center of mass)
        symmetry_score = 0
        if len(person_centroids) >= 2:
            for i in range(len(person_centroids)):
                x1, y1, z1, _ = person_centroids[i]
                # Calculate reflection point across center of mass
                reflection_x = 2*com_x - x1
                reflection_y = 2*com_y - y1
                reflection_z = 2*com_z - z1
                
                # Find distance to closest other performer
                min_reflection_dist = float('inf')
                for j in range(len(person_centroids)):
                    if i != j:
                        x2, y2, z2, _ = person_centroids[j]
                        dist = ((reflection_x-x2)**2 + (reflection_y-y2)**2 + (reflection_z-z2)**2)**0.5
                        min_reflection_dist = min(min_reflection_dist, dist)
                
                # Add to symmetry score - closer reflections mean more symmetry
                if min_reflection_dist != float('inf'):
                    symmetry_score += 1/(1 + min_reflection_dist/100)  # Normalized to 0-1 range
            
            symmetry_score /= len(person_centroids)  # Average symmetry
        
        # Return all calculated metrics
        return {
            'center_of_mass_x': com_x,
            'center_of_mass_y': com_y,
            'center_of_mass_z': com_z,
            'spatial_spread': spatial_spread,
            'avg_performer_distance': avg_dist,
            'max_performer_distance': max_dist,
            'min_performer_distance': min_dist,
            'formation_area': hull_area,
            'symmetry_score': symmetry_score,
            'num_performers': len(people),
            'pairwise_distances': pairwise_distances
        }
        
    def add_frame(self, boxes_3d, frame_number, timestamp):
        """
        Add a frame to the buffer and write to disk if buffer is full
        
        Args:
            boxes_3d: List of 3D box dictionaries containing detection data
            frame_number: Current frame number
            timestamp: Current timestamp
        """
        # Get video height from the first detected person (if available)
        if not hasattr(self, 'video_height'):
            for box in boxes_3d:
                if 'person' in box['class_name'].lower():
                    _, y1, _, y2 = box['bbox_2d']
                    self.video_height = max(y1, y2) * 1.1  # Add some margin
                    print(f"Parquet export: Using video height of {self.video_height} pixels")
                    break
            
            # If no people were found, use a default or try from any object
            if not hasattr(self, 'video_height') and boxes_3d:
                _, y1, _, y2 = boxes_3d[0]['bbox_2d']
                self.video_height = max(y1, y2) * 1.1
                print(f"Parquet export: Using video height of {self.video_height} pixels")
            elif not hasattr(self, 'video_height'):
                self.video_height = 1080  # Default fallback
                print(f"Parquet export: Using default video height of {self.video_height} pixels")
        
        # Calculate group metrics if we have multiple people
        group_metrics = self.calculate_group_metrics(boxes_3d)
        
        # Store one row for group metrics per frame
        if group_metrics:
            # Add center of mass point to parquet
            com_entry = {
                "frame": frame_number,
                "timestamp": timestamp,
                "object_id": -100,  # Special ID for center of mass
                "class": "center_of_mass",
                "keypoint_index": -1,
                "x": float(group_metrics['center_of_mass_x']),
                "y": float(group_metrics['center_of_mass_y']),
                 "depth": float(group_metrics['center_of_mass_z']),
                "confidence": 1.0,
                "spatial_spread": float(group_metrics['spatial_spread']),
                "avg_distance": float(group_metrics['avg_performer_distance']),
                "formation_area": float(group_metrics['formation_area']),
                "symmetry_score": float(group_metrics['symmetry_score']),
                "num_performers": int(group_metrics['num_performers'])
            }
            self.keypoint_data.append(com_entry)
            
            # Add pairwise distance connections as special points
            for dist, id1, id2 in group_metrics.get('pairwise_distances', []):
                connection_entry = {
                    "frame": frame_number,
                    "timestamp": timestamp,
                    "object_id": -200,  # Special ID for connections
                    "class": "performer_connection",
                    "keypoint_index": -1,
                    "x": float(id1),  # Store ID in x
                    "y": float(id2),  # Store ID in y
                    "depth": 0.0,
                    "confidence": 1.0,
                    "distance": float(dist)
                }
                self.keypoint_data.append(connection_entry)
        
        # Process individual detected objects
        for box in boxes_3d:
            # Extract 2D bbox coordinates
            x1, y1, x2, y2 = box['bbox_2d']
            
            # Get object ID (use -1 if not available)
            object_id = box.get('object_id', -1)
            
            # Add keypoints if available
            if 'keypoints' in box and box['keypoints'] is not None:
                for i, kpt in enumerate(box['keypoints']):
                    x, y, conf = kpt
                    depth = box.get('keypoint_depths', [])[i] if i < len(box.get('keypoint_depths', [])) else None
                    
                    # Convert y-coordinate to bottom-left origin (only for parquet export)
                    #y_final = self.video_height - y if self.flip_y else y
                    
                    # Calculate distance to center of mass if available
                    dist_to_com = None
                    if group_metrics:
                        if depth is not None:
                            dist_to_com = ((x - group_metrics['center_of_mass_x'])**2 + 
                                        (group_metrics['center_of_mass_y'])**2 + 
                                        (depth - group_metrics['center_of_mass_z'])**2)**0.5
                        else:
                            dist_to_com = ((x - group_metrics['center_of_mass_x'])**2 + 
                                        (group_metrics['center_of_mass_y'])**2)**0.5
                    
                    # Create row for Parquet
                    flat_kpt = {
                        "frame": frame_number,
                        "timestamp": timestamp,
                        "object_id": object_id,
                        "class": box['class_name'],
                        "keypoint_index": i,
                        "x": float(x),
                        "y": float(y),  # Y-coordinate flipped here
                        "depth": float(depth) if depth is not None else None,
                        "confidence": float(conf),
                        "distance_to_com": float(dist_to_com) if dist_to_com is not None else None
                    }
                    self.keypoint_data.append(flat_kpt)
            
            # If no keypoints are available, add one row for the object centroid
            else:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Convert y-coordinate to bottom-left origin (only for parquet export)
                #center_y_final = self.video_height - center_y if self.flip_y else center_y
                            
                # Calculate distance to center of mass if available
                dist_to_com = None
                if group_metrics and 'depth_value' in box:
                    dist_to_com = ((center_x - group_metrics['center_of_mass_x'])**2 + 
                                (center_y - group_metrics['center_of_mass_y'])**2 + 
                                (box['depth_value'] - group_metrics['center_of_mass_z'])**2)**0.5
                
                self.keypoint_data.append({
                    "frame": frame_number,
                    "timestamp": timestamp,
                    "object_id": object_id,
                    "class": box['class_name'],
                    "keypoint_index": -1,  # -1 indicates object centroid, not a keypoint
                    "x": float(center_x),
                    "y": float(center_y),  # Y-coordinate flipped here
                    "depth": float(box['depth_value']),
                    "confidence": float(box['score']),
                    "distance_to_com": float(dist_to_com) if dist_to_com is not None else None
                })
        
        self.frame_count += 1
        
        # Write to disk if buffer is full to avoid memory issues with large videos
        if len(self.keypoint_data) >= self.buffer_size:
            self.write_partial_parquet()
    
    def write_partial_parquet(self):
        """Write the current buffer to a temporary parquet file"""
        if not self.keypoint_data:
            return
            
        try:
            # Create a temporary file path
            temp_file = f"{self.parquet_output}.part{self.part_count}.parquet"
            
            # Convert to DataFrame and save as Parquet
            df = pd.DataFrame(self.keypoint_data)
            df.to_parquet(temp_file, index=False)
            
            # Add to list of temp files
            self.temp_files.append(temp_file)
            self.part_count += 1
            
            print(f"Written partial data to {temp_file} ({len(self.keypoint_data)} records)")
            
            # Clear buffer
            self.keypoint_data = []
            
        except ImportError:
            print("PyArrow not installed. Install with: pip install pyarrow")
            print("Parquet export skipped.")
        except Exception as e:
            print(f"Error exporting to partial Parquet: {e}")
            
    def finalize(self):
        """Finalize export and merge all partial files"""
        # Write any remaining data
        if self.keypoint_data:
            self.write_partial_parquet()
        
        # If we have multiple partial files, merge them
        if len(self.temp_files) > 1:
            try:
                # Read all partial files into a list of dataframes
                dfs = []
                for temp_file in self.temp_files:
                    df = pd.read_parquet(temp_file)
                    dfs.append(df)
                
                # Concatenate all dataframes
                final_df = pd.concat(dfs, ignore_index=True)
                
                # Write final parquet file
                final_df.to_parquet(self.parquet_output, index=False)
                
                # Delete temporary files
                for temp_file in self.temp_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                
                print(f"Merged {len(self.temp_files)} partial files into final output: {self.parquet_output}")
                print(f"Total records: {len(final_df)}")
                
            except Exception as e:
                print(f"Error merging Parquet files: {e}")
                print("Partial files have been kept for manual merging.")
        
        # If we only have one partial file, just rename it
        elif len(self.temp_files) == 1:
            try:
                os.rename(self.temp_files[0], self.parquet_output)
                print(f"Saved tracking data to {self.parquet_output}")
            except Exception as e:
                print(f"Error finalizing Parquet file: {e}")
        
        print(f"Processed {self.frame_count} frames with tracking data.")

def main():
    """Main function."""
    # Configuration variables (modify these as needed)
    # ===============================================
    
    # Input/Output
    video = "Tonal"  # Path to input video file or webcam index (0 for default camera)
    source = f"/media/M2_disk/roger/sonar/YOLO-3D/videos/in/{video}.MOV" # Path to input video file or webcam index (0 for default camera)
    output_path = f"{video}_output.mp4"  # Path to output video file
    
    # Model settings
    yolo_model_size = "large"  # YOLOv11 model size: "nano", "small", "medium", "large", "extra"
    depth_model_size = "large"  # Depth Anything v2 model size: "small", "base", "large"
    
    # Device settings
    device = 0  # 'cpu' Force CPU for stability
    
    # Detection settings
    conf_threshold = 0.25  # Confidence threshold for object detection
    iou_threshold = 0.45  # IoU threshold for NMS
    classes = None  # Filter by class, e.g., [0, 1, 2] for specific classes, None for all classes
    
    # Feature toggles
    enable_tracking = True  # Enable object tracking
    enable_bev = True  # Enable Bird's Eye View visualization
    enable_pseudo_3d = True  # Enable pseudo-3D visualization
    headless = True  # Enable headless mode (no UI)
    # Camera parameters - simplified approach
    camera_params_file = None  # Path to camera parameters file (None to use default parameters)

    # JSON export settings
    
    json_output = f"videos/out/{video}_tracking_data.parquet"  # Path to output JSON file
    json_buffer_size = 30  # Buffer size (frames before writing to disk)

    # ===============================================
    
    print(f"Using device: {device}")
    
    # Initialize models
    print("Initializing models...")
    try:
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device=device
        )
    except Exception as e:
        print(f"Error initializing object detector: {e}")
        print("Falling back to CPU for object detection")
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device='cpu'
        )
    
    try:
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device=device
        )
    except Exception as e:
        print(f"Error initializing depth estimator: {e}")
        print("Falling back to CPU for depth estimation")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device='cpu'
        )
    
    # Initialize 3D bounding box estimator with default parameters
    # Simplified approach - focus on 2D detection with depth information
    bbox3d_estimator = BBox3DEstimator()
    
    # Initialize Bird's Eye View if enabled
    if enable_bev:
        # Use a scale that works well for the 1-5 meter range
        bev = BirdEyeView(scale=60, size=(300, 300))  # Increased scale to spread objects out
    
    # Open video source
    try:
        if isinstance(source, str) and source.isdigit():
            source = int(source)  # Convert string number to integer for webcam
    except ValueError:
        pass  # Keep as string (for video file)
    
    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:  # Sometimes happens with webcams
        fps = 30
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"
    

    parquet_output = f"./videos/out/{video}_tracking_data.parquet"  # Path to output Parquet file
    tracking_exporter = ParquetExporter(
        parquet_output=parquet_output,
        buffer_size=4000,
        flip_y = False  # Larger buffer for better performance
    )
    print("Starting processing with direct Parquet export...")
    print("Starting processing...")
    
    # Main loop
    while True:
        # Check for key press at the beginning of each loop
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
            print("Exiting program...")
            break
            
        try:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 0)  # Flip frame vertically

            
            # Make copies for different visualizations
            original_frame = frame.copy()
            detection_frame = frame.copy()
            depth_frame = frame.copy()
            result_frame = frame.copy()
            
            # Step 1: Object Detection
            try:
                detection_frame, detections = detector.detect(detection_frame, track=enable_tracking)
            except Exception as e:
                print(f"Error during object detection: {e}")
                detections = []
                cv2.putText(detection_frame, "Detection Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 2: Depth Estimation
            try:
                depth_map = depth_estimator.estimate_depth(original_frame)
                depth_colored = depth_estimator.colorize_depth(depth_map)
            except Exception as e:
                print(f"Error during depth estimation: {e}")
                # Create a dummy depth map
                depth_map = np.zeros((height, width), dtype=np.float32)
                depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(depth_colored, "Depth Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 3: 3D Bounding Box Estimation
            boxes_3d = []
            active_ids = []
            
            for detection in detections:
                try:
                    # Update to handle the new detection format that includes keypoints
                    bbox, score, class_id, obj_id, keypoints = detection
                    
                    # Get class name
                    class_name = detector.get_class_names()[class_id]
                    
                    # Get depth in the region of the bounding box
                    # Try different methods for depth estimation
                    if class_name.lower() in ['person', 'cat', 'dog']:
                        # For people and animals, use the center point depth
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        depth_value = depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                        depth_method = 'center'
                    else:
                        # For other objects, use the median depth in the region
                        depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                        depth_method = 'median'
                    
                    # Process keypoints if available
                    keypoint_depths = []
                    if keypoints is not None:
                        for kpt in keypoints:
                            x, y, conf = kpt
                            if conf > 0.5:  # Only process keypoints with confidence > 0.5
                                # Get depth at keypoint location
                                kpt_depth = depth_estimator.get_depth_at_point(depth_map, int(x), int(y))
                                keypoint_depths.append(kpt_depth)
                            else:
                                keypoint_depths.append(None)
                    
                    # Create a simplified 3D box representation
                    box_3d = {
                        'bbox_2d': bbox,
                        'depth_value': depth_value,
                        'depth_method': depth_method,
                        'class_name': class_name,
                        'object_id': obj_id,
                        'score': score,
                        'keypoints': keypoints,
                        'keypoint_depths': keypoint_depths
                    }
                    
                    boxes_3d.append(box_3d)
                    
                    # Keep track of active IDs for tracker cleanup
                    if obj_id is not None:
                        active_ids.append(obj_id)
                except ValueError as e:
                    print(f"Detection format error: {e}")
                    print(f"Detection data: {detection}")
                    continue
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            
            # Clean up trackers for objects that are no longer
            # 
            #  detected
            bbox3d_estimator.cleanup_trackers(active_ids)
            timestamp = time.time() - start_time  # Time since start of processing
            tracking_exporter.add_frame(boxes_3d, frame_count, timestamp)
            # Step 4: Visualization
            # Draw boxes on the result frame
            for box_3d in boxes_3d:
                try:
                    # Determine color based on class
                    class_name = box_3d['class_name'].lower()
                    if 'car' in class_name or 'vehicle' in class_name:
                        color = (0, 0, 255)  # Red
                    elif 'person' in class_name:
                        color = (0, 255, 0)  # Green
                    elif 'bicycle' in class_name or 'motorcycle' in class_name:
                        color = (255, 0, 0)  # Blue
                    elif 'potted plant' in class_name or 'plant' in class_name:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (255, 255, 255)  # White
                    
                    # Draw box with depth information
                    result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                    
                    # Draw keypoints and skeleton if available
                    if 'keypoints' in box_3d and box_3d['keypoints'] is not None:
                        keypoints = box_3d['keypoints']
                        
                        # Draw individual keypoints with depth information
                        for i, kpt in enumerate(keypoints):
                            x, y, conf = kpt
                            if conf > 0.5:  # Only draw keypoints with confidence > 0.5
                                # Get depth value for this keypoint
                                kpt_depth = box_3d.get('keypoint_depths', [])[i] if i < len(box_3d.get('keypoint_depths', [])) else None
                                
                                # Use depth to determine circle color (red->yellow->green from near to far)
                                if kpt_depth is not None:
                                    # Normalize depth within 0-5 meter range
                                    normalized_depth = min(1.0, max(0.0, kpt_depth / 5.0))
                                    # Create color: red (0,0,255) to green (0,255,0)
                                    kpt_color = (0, int(255 * normalized_depth), int(255 * (1-normalized_depth)))
                                else:
                                    kpt_color = (0, 255, 0)  # Default green
                                
                                # Draw keypoint circle
                                cv2.circle(result_frame, (int(x), int(y)), 5, kpt_color, -1)
                                
                                # Optionally display depth next to important keypoints (e.g., head)
                                if i == 0 and kpt_depth is not None:  # Assuming 0 is the head keypoint
                                    cv2.putText(result_frame, f"{kpt_depth:.2f}m", 
                                            (int(x) + 5, int(y) - 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, kpt_color, 1)
                        
                        # Draw skeleton if it's a person
                        if 'person' in class_name:
                            # Use the draw_skeleton method from your detector
                            result_frame = detector.draw_skeleton(result_frame, np.array(keypoints))
                except Exception as e:
                    print(f"Error drawing object: {e}")
                    continue

            
            # Draw Bird's Eye View if enabled
            if enable_bev:
                try:
                    # Reset BEV and draw objects
                    bev.reset()
                    for box_3d in boxes_3d:
                        bev.draw_box(box_3d)
                    bev_image = bev.get_image()
                    
                    # Resize BEV image to fit in the corner of the result frame
                    bev_height = height // 4  # Reduced from height/3 to height/4 for better fit
                    bev_width = bev_height
                    
                    # Ensure dimensions are valid
                    if bev_height > 0 and bev_width > 0:
                        # Resize BEV image
                        bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
                        
                        # Create a region of interest in the result frame
                        roi = result_frame[height - bev_height:height, 0:bev_width]
                        
                        # Simple overlay - just copy the BEV image to the ROI
                        result_frame[height - bev_height:height, 0:bev_width] = bev_resized
                        
                        # Add a border around the BEV visualization
                        cv2.rectangle(result_frame, 
                                     (0, height - bev_height), 
                                     (bev_width, height), 
                                     (255, 255, 255), 1)
                        
                        # Add a title to the BEV visualization
                        cv2.putText(result_frame, "Bird's Eye View", 
                                   (10, height - bev_height + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Error drawing BEV: {e}")
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 10 == 0:  # Update FPS every 10 frames
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps_value = frame_count / elapsed_time
                fps_display = f"FPS: {fps_value:.1f}"
            
            # Add FPS and device info to the result frame
            cv2.putText(result_frame, f"{fps_display} | Device: {device}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            
            # Add depth map to the corner of the result frame
            try:
                depth_height = height // 4
                depth_width = depth_height * width // height
                depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
                result_frame[0:depth_height, 0:depth_width] = depth_resized
            except Exception as e:
                print(f"Error adding depth map to result: {e}")
            
            # Write frame to output video
            out.write(result_frame)
            
            if not headless:
                # Display frames only if not in headless mode
                cv2.imshow("3D Object Detection", result_frame)
                cv2.imshow("Depth Map", depth_colored)
                cv2.imshow("Object Detection", detection_frame)
                # Check for key press at the end of the loop
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                    print("Exiting program...")
                    break
            else:
                # For headless mode, we need a different way to handle early termination
                # This just prints progress but doesn't check for key presses
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames ({fps_display})")
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Also check for key press during exception handling
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                print("Exiting program...")
                break
            continue
    
    # Clean up
    print("Cleaning up resources...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    tracking_exporter.finalize()
    print(f"Processing complete. Output saved to {output_path}")
    print(f"Tracking data saved to {json_output} for Houdini import")




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        # Clean up OpenCV windows
        cv2.destroyAllWindows() 