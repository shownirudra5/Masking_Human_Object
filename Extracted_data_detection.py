import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO
from PIL import Image
from collections import defaultdict

# --- 1. Configuration ---
RGB_IMAGES_DIR = os.path.join("C:", os.sep, "Users", "stitli", "Desktop", "ZED",
                              "Extracted_Data", "Activity1", "person2",
                              "ZED 21283893", "HD1080_SN29755398_17-52-10_Maciek1", "rgb")
OUTPUT_VIDEO_PATH = "HOI_detection_output1.avi"
FPS = 30.0

# COCO Class IDs for relevant objects (furniture and common items)
PERSON_CLASS_ID = 0
RELEVANT_OBJECTS = {
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 73: 'book', 74: 'clock', 75: 'vase',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 76: 'scissors', 84: 'screwdriver'
}

# Activity keywords based on pose and object interaction
ACTIVITIES = {
    'standing': 'Standing',
    'sitting': 'Sitting',
    'reaching': 'Reaching for object',
    'interacting': 'Interacting with object',
    'walking': 'Walking',
    'idle': 'Standing idle'
}

# Color mapping for visualization (BGR format)
COLOR_MAP = {
    PERSON_CLASS_ID: (255, 0, 0),    # Blue for Human
}
# Generate distinct colors for different object classes
np.random.seed(42)
for class_id in RELEVANT_OBJECTS.keys():
    COLOR_MAP[class_id] = tuple(np.random.randint(0, 255, 3).tolist())

# Interaction detection threshold (IoU-based)
INTERACTION_THRESHOLD = 0.05  # Reduced threshold for nearby objects

# --- ROBUST IMAGE READER (SIMPLIFIED) ---
def read_image_robust(image_path):
    """
    Reads an image using a robust method (cv2.imdecode) that handles
    special characters and spaces in file paths on Windows.
    Uses PIL as a fallback.
    Returns a contiguous BGR NumPy array.
    """
    
    # Method 1: Try the robust OpenCV method first
   
    try:
        with open(image_path, 'rb') as f:
            raw_bytes = f.read()

        if not raw_bytes:
            print(f"   {os.path.basename(image_path)} is 0 bytes (empty).")
            return None
            
        # Create a 1D numpy array from the raw bytes
        n = np.frombuffer(raw_bytes, dtype=np.uint8)
        
        # Decode as a 3-CHANNEL BGR image (this is the key change)
        frame = cv2.imdecode(n, cv2.IMREAD_COLOR) 
        
        if frame is not None and frame.size > 0:
            return np.ascontiguousarray(frame)
        else:
            # imdecode failed, fall through to PIL
            print(f" CV2 imdecode failed silently for {os.path.basename(image_path)}. Trying PIL.")
            pass

    except Exception as e_cv:
        print(f" CV2 method failed for {os.path.basename(image_path)}: {e_cv}")
        pass

    # Method 2: Fallback to PIL (Pillow)
    try:
        pil_img = Image.open(image_path).convert('RGB')
        frame_rgb = np.array(pil_img, dtype=np.uint8)
        
        # Convert RGB (PIL) to BGR (OpenCV)
        # OLD: frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # NEW: Convert RGB to BGR using pure numpy slicing
        # This avoids the cv2.cvtColor error
        frame_bgr = frame_rgb[..., ::-1]

        if frame_bgr is not None and frame_bgr.size > 0:
            print(f"Loaded {os.path.basename(image_path)} using PIL fallback (numpy convert).")
            return np.ascontiguousarray(frame_bgr)
            
    except Exception as e_pil:
        # Both methods have now failed.
        print(f"PIL method ALSO failed for {os.path.basename(image_path)}: {e_pil}")
        pass
        
    # If we get here, both methods failed.
    print(f" FAILED to load {os.path.basename(image_path)} using all methods.")
    return None

# --- VIDEO SETUP ---
def setup_video_writer(image_files):
    """Find first valid frame and set up video writer."""
    first_frame = None
    
    print(f"Searching for first valid frame...")
    
    for i in range(min(len(image_files), 100)):
        first_frame = read_image_robust(image_files[i])
        
        if first_frame is not None and first_frame.size > 0:
            print(f" Found valid frame at index {i}: {os.path.basename(image_files[i])}")
            print(f"  Frame shape: {first_frame.shape}")
            break
    
    if first_frame is None:
        raise FileNotFoundError("Failed to load any valid image.")
        
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (width, height))
    
    if not out.isOpened():
        raise RuntimeError("Failed to open video writer.")
    
    return out, width, height

# --- INTERACTION DETECTION ---
def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_distance(box1, box2):
    """Calculate normalized distance between centers of two boxes."""
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    
    distance = np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
    
    # Normalize by image diagonal
    img_diagonal = np.sqrt((box1[2] - box1[0])**2 + (box1[3] - box1[1])**2)
    return distance / img_diagonal if img_diagonal > 0 else float('inf')

def detect_interactions(boxes, classes):
    """Detect human-object interactions and infer activity."""
    interactions = []
    
    person_indices = [i for i, cls in enumerate(classes) if cls == PERSON_CLASS_ID]
    object_indices = [i for i, cls in enumerate(classes) if cls in RELEVANT_OBJECTS.keys()]
    
    for person_idx in person_indices:
        person_box = boxes[person_idx]
        person_height = person_box[3] - person_box[1]
        person_width = person_box[2] - person_box[0]
        
        # Analyze person's aspect ratio and position for activity
        aspect_ratio = person_height / person_width if person_width > 0 else 0
        
        nearby_objects = []
        
        for obj_idx in object_indices:
            obj_box = boxes[obj_idx]
            obj_class = int(classes[obj_idx])
            
            # Calculate IoU and distance
            iou = calculate_iou(person_box, obj_box)
            distance = calculate_distance(person_box, obj_box)
            
            # Detect interaction if IoU > threshold or distance is small
            if iou > INTERACTION_THRESHOLD or distance < 0.3:
                nearby_objects.append({
                    'object_idx': obj_idx,
                    'object_class': obj_class,
                    'object_name': RELEVANT_OBJECTS[obj_class],
                    'iou': iou,
                    'distance': distance
                })
        
        # Infer activity based on nearby objects and person geometry
        activity = infer_activity(person_box, nearby_objects, aspect_ratio)
        
        if nearby_objects or activity != 'idle':
            interactions.append({
                'person_idx': person_idx,
                'person_box': person_box,
                'nearby_objects': nearby_objects,
                'activity': activity,
                'aspect_ratio': aspect_ratio
            })
    
    return interactions

def infer_activity(person_box, nearby_objects, aspect_ratio):
    """Infer human activity based on context."""
    # If person is interacting with objects
    if nearby_objects:
        obj_names = [obj['object_name'] for obj in nearby_objects]
        
        # Check for specific furniture interactions
        if any(name in ['chair', 'couch', 'bed'] for name in obj_names):
            return 'Sitting/Resting'
        elif 'dining table' in obj_names:
            return 'Interacting with table'
        elif any(name in ['laptop', 'keyboard', 'mouse', 'cell phone'] for name in obj_names):
            return 'Using device'
        elif any(name in ['bottle', 'cup', 'bowl'] for name in obj_names):
            return 'Eating/Drinking'
        else:
            return f'Interacting with {obj_names[0]}'
    
    # Infer from aspect ratio (standing vs sitting)
    if aspect_ratio > 2.0:
        return 'Standing'
    elif aspect_ratio < 1.5:
        return 'Sitting/Crouching'
    else:
        return 'Standing'

# --- MASKING AND VISUALIZATION ---
def apply_mask_overlay(frame, results):
    """Apply color masks based on YOLO segmentation results."""
    masked_frame = frame.copy()
    
    if results and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for i, mask in enumerate(masks):
            class_id = classes[i]
            color = COLOR_MAP.get(class_id, (0, 255, 0))

            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            mask_ocv = (mask_resized * 255).astype(np.uint8)

            color_layer = np.full(masked_frame.shape, color, dtype=np.uint8)
            masked_color_layer = cv2.bitwise_and(color_layer, color_layer, mask=mask_ocv)

            alpha = 0.5 
            cv2.addWeighted(masked_color_layer, alpha, masked_frame, 1 - alpha, 0, masked_frame)
            
    return masked_frame

def draw_interactions(frame, results, interactions):
    """Draw bounding boxes, labels, and activity information."""
    if not results or results[0].boxes is None:
        return frame
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()
    
    # Draw all detected objects
    for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        color = COLOR_MAP.get(cls, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if cls == PERSON_CLASS_ID:
            label = f"Person {conf:.2f}"
        elif cls in RELEVANT_OBJECTS:
            label = f"{RELEVANT_OBJECTS[cls]} {conf:.2f}"
        else:
            continue  # Skip irrelevant objects
        
        # Background for text
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw activity information for each person
    for interaction in interactions:
        person_box = boxes[interaction['person_idx']]
        x1, y1, x2, y2 = map(int, person_box)
        
        # Draw activity label above person
        activity_text = f"Activity: {interaction['activity']}"
        (w, h), _ = cv2.getTextSize(activity_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Background rectangle
        cv2.rectangle(frame, (x1, y1 - 50), (x1 + w + 10, y1 - 25), (0, 255, 255), -1)
        cv2.putText(frame, activity_text, (x1 + 5, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw lines to nearby objects
        person_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        for obj in interaction['nearby_objects']:
            obj_box = boxes[obj['object_idx']]
            ox1, oy1, ox2, oy2 = map(int, obj_box)
            obj_center = (int((ox1 + ox2) / 2), int((oy1 + oy2) / 2))
            
            # Draw interaction line
            cv2.line(frame, person_center, obj_center, (0, 255, 255), 2)
            
            # Draw object name near the line
            mid_point = (int((person_center[0] + obj_center[0]) / 2),
                         int((person_center[1] + obj_center[1]) / 2))
            cv2.putText(frame, obj['object_name'], mid_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return frame

# --- MAIN PROCESSING ---
def process_images_to_video():
    """Main function to process images and create video."""
    
    # Get all image files
    # MODIFICATION: Re-enabled .png files
    print("Searching for .jpg and .png files...")
    image_files = sorted(
        glob.glob(os.path.join(RGB_IMAGES_DIR, "*.png")) +
        glob.glob(os.path.join(RGB_IMAGES_DIR, "*.jpg"))
    )
    
    if not image_files:
        print(f"Error: No image files found in {RGB_IMAGES_DIR}")
        return

    print(f"Found {len(image_files)} image files")
    
    # Setup video writer
    try:
        out, width, height = setup_video_writer(image_files)
        print(f"Video writer initialized: {width}x{height} @ {FPS} FPS")
    except Exception as e:
        print(f"Setup failed: {e}")
        return
    
    # Initialize YOLO model
    print("Initializing YOLO model...")
    try:
        model = YOLO('yolov8n-seg.pt')
        print(" YOLO model loaded")
    except Exception as e:
        print(f"Error initializing YOLO: {e}")
        out.release()
        return

    # Process frames
    print(f"\nProcessing {len(image_files)} frames...")
    print(f"Output: {OUTPUT_VIDEO_PATH}\n")
    
    processed_count = 0
    skipped_count = 0
    activity_stats = defaultdict(int)
    
    for i, image_path in enumerate(image_files):
        # Progress update
        if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
            print(f"Processing: {i + 1}/{len(image_files)} "
                  f"(Processed: {processed_count}, Skipped: {skipped_count})")

        # Read frame
        frame = read_image_robust(image_path)
        
        if frame is None:
            skipped_count += 1
            continue

        # Run YOLO detection
        try:
            results = model(frame, verbose=False)
            
            # Apply masks
            masked_frame = apply_mask_overlay(frame, results)
            
            # Detect interactions and activities
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                interactions = detect_interactions(boxes, classes)
                
                # Update statistics
                for interaction in interactions:
                    activity_stats[interaction['activity']] += 1
                
                # Draw interactions and activities
                masked_frame = draw_interactions(masked_frame, results, interactions)
            
            # Write to video
            out.write(masked_frame)
            processed_count += 1
        except Exception as e:
            print(f"  ⚠ Error processing frame {i}: {e}")
            skipped_count += 1
            continue

    # Cleanup
    out.release()
    
    print(f"\n{'='*50}")
    print(f"✓ Processing complete!")
    print(f"  Processed frames: {processed_count}")
    print(f"  Skipped frames: {skipped_count}")
    print(f"  Output: {os.path.abspath(OUTPUT_VIDEO_PATH)}")
    print(f"\n{'='*50}")
    print(f"Human Activity Statistics:")
    print(f"{'='*50}")
    if activity_stats:
        for activity, count in sorted(activity_stats.items(),
                                      key=lambda x: x[1], reverse=True):
            print(f"  {activity}: {count} frames")
    else:
        print("  No activities detected")
    print(f"{'='*50}")

# --- RUN ---
if __name__ == "__main__":
    process_images_to_video()