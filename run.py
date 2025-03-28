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

def main():
    """Main function."""
    # Configuration variables (modify these as needed)
    # ===============================================
    
    # Input/Output
    source = 0 # Path to input video file or webcam index (0 for default camera)
    output_path = "output.mp4"  # Path to output video file
    
    # Model settings
    yolo_model_size = "nano"  # YOLOv11 model size: "nano", "small", "medium", "large", "extra"
    depth_model_size = "small"  # Depth Anything v2 model size: "small", "base", "large"
    
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
            
            # Clean up trackers for objects that are no longer detected
            bbox3d_estimator.cleanup_trackers(active_ids)
            
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
    
    print(f"Processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        # Clean up OpenCV windows
        cv2.destroyAllWindows() 