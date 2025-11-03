# This script is designed for a Python Notebook (like Jupyter or Colab).
# Copy and paste each "CELL" block into a new cell in your notebook and run them in order.

# ---
# --- CELL 1: Imports ---
# ---
# Run this cell first to import all necessary libraries
import json
import math
import os
import cv2
from ultralytics import YOLO
import numpy as np # Often needed with cv2/yolo

print("Cell 1: Imports complete.")

# ---
# --- CELL 2: CONFIGURATION (EDIT THIS CELL) ---
# ---
# This is the main cell you need to edit.
# Set your class map and file paths here.

# 1. DEFINE YOUR CLASS MAP
# This is the CORRECTED map based on the 11-item list you provided.
CLASS_MAP = {
    0: 'MMs_peanut',
    1: 'MMs_regular',
    2: 'airheads',
    3: 'gummy_worms',
    4: 'milky_way',
    5: 'nerds',
    6: 'skittles',
    7: 'snickers',
    8: 'starbust',
    9: 'three_musketeers',
    10: 'twizzlers'
}

# 2. DEFINE YOUR FILE PATHS
# Update these paths to match your notebook's environment (e.g., /content/...)
VIDEO_PATH = r"C:\Users\lenovo\Downloads\BTP\Video_Ready_After_Top_View.mp4"
MODEL_PATH = r"C:\Users\lenovo\Downloads\BTP\my_model\my_model.pt"
INVENTORY_FILE_PATH = r"C:\Users\lenovo\Downloads\BTP\inventory.json"

print("Cell 2: Configuration loaded.")
print(f"  Class Map (0-10): {list(CLASS_MAP.values())}")
print(f"  Model Path: {MODEL_PATH}")
print(f"  Video Path: {VIDEO_PATH}")
print(f"  Inventory File: {INVENTORY_FILE_PATH}")


# ---
# --- CELL 3: Core Logic Functions (NEW R-to-L Add Logic) ---
# ---
# Run this cell to define all the analysis functions.
# You don't need to edit this.

def load_inventory(filepath):
    """Loads inventory from a JSON file, or returns a default if it doesn't exist."""
    print(f"Loading inventory from {filepath}...")
    if not os.path.exists(filepath):
        print("  Inventory file not found. Starting with an empty inventory.")
        return {}
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"  Error reading {filepath}. Starting with empty inventory.")
        return {}

def save_inventory(filepath, inventory):
    """Saves the inventory to a JSON file."""
    print(f"Saving updated inventory to {filepath}...")
    try:
        with open(filepath, 'w') as f:
            json.dump(inventory, f, indent=2)
        print("  Save successful.")
    except Exception as e:
        print(f"  Error saving inventory: {e}")

def calculate_centroid(bbox):
    """
    Calculates the center (cx, cy) of a bounding box.
    Bbox format: [xmin, ymin, xmax, ymax] (from YOLOv8)
    """
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    return cx, cy

def find_best_match(initial_det, final_detections_list, matched_final_indices):
    """
    Finds the closest object of the *same class* in the final detections.
    This is a simple "tracking-by-detection" approach.
    Returns the match, its index, and the distance.
    (Threshold logic has been removed from this function.)
    """
    initial_centroid = calculate_centroid(initial_det['bbox'])
    initial_class_id = initial_det['class_id']
    
    best_match = None
    best_match_index = -1
    min_dist = float('inf')
    
    for i, final_det in enumerate(final_detections_list):
        # Skip if different class or already matched
        if final_det['class_id'] != initial_class_id or i in matched_final_indices:
            continue
            
        final_centroid = calculate_centroid(final_det['bbox'])
        dist = math.dist(initial_centroid, final_centroid)
        
        if dist < min_dist:
            min_dist = dist
            best_match = final_det
            best_match_index = i
            
    return best_match, best_match_index, min_dist

def analyze_inventory_changes(initial_detections, final_detections, config, class_map):
    """
    Analyzes changes between two frames.
    
    NEW LOGIC:
    - Item moves Left-to-Right = SALE (-1).
    - Item moves Right-to-Left = ADD (+1).
    - Item DISAPPEARS = SALE (-1).
    - Item APPEARS = SALE (-1).
    """
    
    print("--- Analyzing Inventory Changes (Complex) ---")
    
    # Configuration parameters
    print("  Matching items based on closest distance of the same class.")
    
    inventory_changes = {} # e.g., {'snickers': -1, 'airheads': +1}
    
    final_objects_list = [det for det in final_detections]
    matched_final_indices = set()

    # ---
    # Pass 1: Check for items that MOVED or DISAPPEARED
    # ---
    print("\n[Pass 1: Checking for MOVED or DISAPPEARED items]")
    for initial_det in initial_detections:
        
        initial_label = class_map.get(initial_det['class_id'], f"ID:{initial_det['class_id']}")
        
        # Find the best match in the final frame
        best_match, best_match_index, min_dist = find_best_match(
            initial_det, final_objects_list, matched_final_indices
        )
        
        # Case 1: A match was found (object moved or stayed)
        if best_match is not None:
            matched_final_indices.add(best_match_index)
            
            # --- NEW Left-to-Right SALE / Right-to-Left ADD LOGIC ---
            cx_initial, _ = calculate_centroid(initial_det['bbox'])
            cx_final, _ = calculate_centroid(best_match['bbox'])
            
            if cx_final > cx_initial:
                print(f"  [SALE]: '{initial_label}' moved Left-to-Right (X: {cx_initial:.0f} -> {cx_final:.0f}).")
                inventory_changes[initial_label] = inventory_changes.get(initial_label, 0) - 1
            elif cx_final < cx_initial:
                print(f"  [ADD]: '{initial_label}' moved Right-to-Left (X: {cx_initial:.0f} -> {cx_final:.0f}).")
                inventory_changes[initial_label] = inventory_changes.get(initial_label, 0) + 1
            else:
                print(f"  [NO CHANGE]: '{initial_label}' stayed still (dist: {min_dist:.1f}).")
            # --- END NEW LOGIC ---
        
        # Case 2: No match found (object was removed)
        else:
            print(f"  [SALE]: '{initial_label}' DISAPPEARED from frame.")
            inventory_changes[initial_label] = inventory_changes.get(initial_label, 0) - 1

    # ---
    # Pass 2: Check for items that APPEARED
    # ---
    print("\n[Pass 2: Checking for NEWLY APPEARED items (Sale)]")
    for i, final_det in enumerate(final_objects_list):
        if i not in matched_final_indices:
            final_label = class_map.get(final_det['class_id'], f"ID:{final_det['class_id']}")
            print(f"  [SALE]: '{final_label}' APPEARED in frame.")
            inventory_changes[final_label] = inventory_changes.get(final_label, 0) - 1
                
    print("--- Analysis Complete ---")
    return inventory_changes

def apply_inventory_updates(current_inventory, inventory_changes):
    """
    Updates the inventory counts based on the detected changes.
    """
    print("\n--- Updating Inventory ---")
    
    new_inventory = current_inventory.copy()
    
    if not inventory_changes:
        print("No inventory changes detected.")
        return new_inventory

    for item, change_amount in inventory_changes.items():
        if item not in new_inventory:
            print(f"  Warning: Item '{item}' not in inventory. Initializing to 0.")
            new_inventory[item] = 0
            
        # Handle both ADD and SALE
        if change_amount > 0:
             print(f"  Adding {change_amount} to '{item}' (Restock).")
        elif change_amount < 0:
            print(f"  Subtracting {abs(change_amount)} from '{item}' (Sale).")
            
        new_inventory[item] += change_amount
            
    return new_inventory

print("Cell 3: Logic functions defined.")


# ---
# --- CELL 4: Video & YOLO Helper Functions ---
# ---
# Run this cell to define the video processing functions.
# You don't need to edit this.

def parse_yolo_results(results, class_map):
    """Converts YOLO results into the format our logic function expects."""
    detections = []
    boxes = results.boxes
    print(f"    Raw YOLO output: {len(boxes)} detections.")
    
    # --- DEBUG: Print what YOLO found by Class ID ---
    found_ids = [int(boxes.cls[i]) for i in range(len(boxes))]
    print(f"    Found Class IDs: {found_ids}")
    # --- END DEBUG ---
    
    for i in range(len(boxes)):
        class_id = int(boxes.cls[i])
        if class_id not in class_map:
            print(f"    Warning: Skipping unknown class ID: {class_id}")
            continue
            
        detections.append({
            'class_id': class_id,
            'bbox': boxes.xyxy[i].tolist(),
            'confidence': float(boxes.conf[i])
        })
    return detections

def get_video_frames(video_path):
    """
    Grabs frames from a video based on new timing:
    - Initial Frame: 2 seconds after start
    - Final Frame: 2 seconds before end
    """
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None, None, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Video FPS is 0. Defaulting to 30.")
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps
    print(f"  Video Info: {frame_count} frames, {fps:.2f} FPS, {duration_sec:.2f}s duration")

    # --- NEW TIMING LOGIC ---
    # 1. Get initial frame (sample at 2 seconds)
    initial_frame_time_msec = 2000  # 2 seconds
    
    # 2. Get final frame (sample 2 seconds from end)
    final_frame_time_msec = (duration_sec - 2) * 1000

    # Safety checks for short videos
    if duration_sec <= 4.5: # Increased threshold slightly
        print("  Warning: Video is <= 4.5 seconds. Using 1st and last frame.")
        initial_frame_time_msec = 0
        final_frame_time_msec = (duration_sec - (1/fps)) * 1000 # Go to almost the end
    elif final_frame_time_msec <= initial_frame_time_msec:
         print("  Warning: Video is too short for 2s/2s split. Using 1/3 and 2/3 marks.")
         initial_frame_time_msec = (duration_sec / 3) * 1000
         final_frame_time_msec = (duration_sec * 2 / 3) * 1000
    # --- END NEW TIMING LOGIC ---

    print(f"  Sampling initial frame at {initial_frame_time_msec/1000:.2f}s")
    cap.set(cv2.CAP_PROP_POS_MSEC, initial_frame_time_msec)
    ret, initial_frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame.")
        cap.release()
        return None, None, 0

    print(f"  Sampling final frame at {final_frame_time_msec/1000:.2f}s")
    cap.set(cv2.CAP_PROP_POS_MSEC, final_frame_time_msec)
    ret, final_frame = cap.read()
    if not ret:
        print("  Warning: Could not read final frame. Using initial frame as final.")
        final_frame = initial_frame.copy() # Use a copy

    image_width = initial_frame.shape[1]
    cap.release()
    print("  Successfully sampled 'initial' and 'final' frames.")
    return initial_frame, final_frame, image_width

print("Cell 4: Helper functions defined.")



# ---
# --- CELL 5: MAIN EXECUTION ---
# ---
# This is the final cell. Running this will:
# 1. Load your model and inventory.
# 2. Process your video.
# 3. Run YOLO on the start/end frames.
# 4. Analyze the changes.
# 5. Save the new inventory counts.
#
# You can run this cell every time you want to process a new video.
# (It will use the paths you set in CELL 2)
# ---

print("="*40)
print("STARTING VIDEO INVENTORY PROCESSOR")
print("="*40)

# 1. Load Model
print(f"Loading YOLO model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("  Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load model. Check path. {e}")
    # In a notebook, we might not want to exit(), so just print the error
    raise e

# 2. Load Inventory
current_inventory = load_inventory(INVENTORY_FILE_PATH)
print("\n--- CURRENT INVENTORY ---")
print(json.dumps(current_inventory, indent=2) if current_inventory else "{} (Empty)")

# 3. Get Video Frames
initial_frame, final_frame, image_width = get_video_frames(VIDEO_PATH)

if initial_frame is not None:
    # 4. Run YOLO on frames
    print("\nRunning YOLO on 'initial' frame...")
    # Set confidence threshold to catch more objects,
    # as items might be partially obscured
    initial_results = model.predict(initial_frame, verbose=False, conf=0.25)[0]
    initial_detections = parse_yolo_results(initial_results, CLASS_MAP)
    print(f"  -> Found {len(initial_detections)} known objects.")

    print("\nRunning YOLO on 'final' frame...")
    final_results = model.predict(final_frame, verbose=False, conf=0.25)[0]
    final_detections = parse_yolo_results(final_results, CLASS_MAP)
    print(f"  -> Found {len(final_detections)} known objects.")

    # 5. Run Analysis
    analysis_config = {
        "image_width": image_width
        # match_threshold_pixels has been removed.
    }

    inventory_changes = analyze_inventory_changes(
        initial_detections,
        final_detections,
        analysis_config,
        CLASS_MAP
    )

    # 6. Apply Updates
    if inventory_changes:
        print("\n--- FINAL INVENTORY CHANGES ---")
        print(json.dumps(inventory_changes, indent=2))

        updated_inventory = apply_inventory_updates(current_inventory, inventory_changes)

        print("\n--- FINAL UPDATED INVENTORY ---")
        print(json.dumps(updated_inventory, indent=2))

        # 7. Save Updates
        save_inventory(INVENTORY_FILE_PATH, updated_inventory)
    else:
        print("\nNo inventory changes to report.")
        print("--- FINAL INVENTORY (Unchanged) ---")
        print(json.dumps(current_inventory, indent=2))

else:
    print("FATAL ERROR: Could not process video. Exiting.")

print("\n" + "="*40)
print("PROCESS COMPLETE")
print("="*40)