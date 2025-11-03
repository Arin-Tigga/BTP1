import json
import math
import os
from ultralytics.engine.results import Boxes

# ---
# --- 1. CONFIGURATION: CLASS MAP ---
# ---
# This map MUST match your model's .yaml file
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

# ---
# --- 2. Core Logic Functions ---
# ---

def calculate_centroid(bbox):
    """
    Calculates the center (cx, cy) of a bounding box.
    Bbox format: [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    return cx, cy

def find_best_match(initial_det, final_detections_list, matched_final_indices):
    """
    Finds the closest object of the *same class* in the final detections.
    Returns the match, its index, and the distance.
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
    Analyzes changes between two frames based on your custom logic.
    
    LOGIC:
    - Item moves Left-to-Right = SALE (-1).
    - Item moves Right-to-Left = ADD (+1).
    - Item DISAPPEARS = SALE (-1).
    - Item APPEARS = SALE (-1).
    """
    
    print("--- Analyzing Inventory Changes ---")
    
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
            
            # --- L-to-R SALE / R-to-L ADD LOGIC ---
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
            # --- END LOGIC ---
        
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

# ---
# --- 3. File and Data Helper Functions ---
# ---

def load_inventory(filepath):
    """Loads inventory from a JSON file, or returns an empty dict if it doesn't exist."""
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

def parse_yolo_results(results_object, class_map):
    """Converts a single YOLO results object into the format our logic function expects."""
    detections = []
    
    if results_object.boxes is None:
        print("    No boxes found in YOLO results.")
        return []
        
    boxes: Boxes = results_object.boxes
    
    print(f"    Raw YOLO output: {len(boxes)} detections.")
    
    # --- DEBUG: Print what YOLO found by Class ID ---
    found_ids = []
    if boxes.cls is not None:
         found_ids = [int(c) for c in boxes.cls]
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