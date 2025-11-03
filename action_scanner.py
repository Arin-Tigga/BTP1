import cv2
import time
import json
import sys
import threading
import io
import socketserver
import http.server
from ultralytics import YOLO

# --- MJPEG server globals (populated if --mjpeg-port is passed) ---
_latest_jpeg = None
_latest_jpeg_lock = threading.Lock()


class _MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/stream':
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.end_headers()

        try:
            while True:
                with _latest_jpeg_lock:
                    frame = _latest_jpeg
                if frame is None:
                    time.sleep(0.05)
                    continue

                self.wfile.write(b"--FRAME\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(0.05)
        except Exception:
            # client disconnected
            return


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def _start_mjpeg_server(port: int):
    server = _ThreadingHTTPServer(('0.0.0.0', port), _MJPEGHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"MJPEG stream available at http://localhost:{port}/stream")
    return server

# ---
# --- 1. IMPORT YOUR LOGIC MODULE ---
# ---
try:
    import inventory_logic as logic
except ImportError:
    print("FATAL ERROR: Could not find 'inventory_logic.py'.")
    print("Please make sure both 'action_scanner.py' and 'inventory_logic.py' are in the same folder.")
    exit()

# ---
# --- 2. CONFIGURATION ---
# ---
MODEL_PATH = r"C:\Users\suran\OneDrive\Desktop\VScode_WorkSpaces\Intermediate_WS_projects\BTP1\my_model\my_model.pt"        # <-- SET PATH to your YOLO model
INVENTORY_FILE_PATH = r"C:\Users\suran\OneDrive\Desktop\VScode_WorkSpaces\Intermediate_WS_projects\BTP1\inventory.json"      # <-- SET PATH for your inventory file
WEBCAM_ID = 0                                      # <-- 0 is usually the default webcam
CONFIDENCE_THRESHOLD = 0.5                         # <-- Adjust as needed
SCAN_DURATION_SEC = 10                             # <-- Duration of the scan window

# ---
# --- 3. MAIN EXECUTION ---
# ---
def main():
    print("="*40)
    print("STARTING ACTION SCANNER")
    print(f"Press 's' to start a {SCAN_DURATION_SEC}-second scan.")
    print("Press 'q' in the video window to quit.")
    print("="*40)

    # 1. Load Model
    print(f"Loading YOLO model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"FATAL ERROR: Could not load model. Check path. {e}")
        exit()

    # 2. Load Inventory
    inventory = logic.load_inventory(INVENTORY_FILE_PATH)
    print("\n--- CURRENT INVENTORY ---")
    print(json.dumps(inventory, indent=2) if inventory else "{} (Empty)")
    print("="*40)

    # 3. Open Webcam
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print(f"FATAL ERROR: Could not open webcam (ID: {WEBCAM_ID}).")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # State machine variables
    current_state = "idle"  # Can be "idle" or "scanning"
    scan_start_time = None
    initial_frame = None

    # 4. Start Real-Time Loop
    while True:
        success, frame = cap.read()
        if not success:
            print("Webcam frame read failed. Exiting.")
            break

        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and current_state == "idle":
            print("\n" + "="*20)
            print(f"STARTING {SCAN_DURATION_SEC}-SECOND SCAN...")
            current_state = "scanning"
            scan_start_time = time.time()
            # Save the "before" shot!
            initial_frame = frame.copy()
            print("  'Initial' frame captured.")

        
        # ---
        # State: IDLE (Waiting for user)
        # ---
        if current_state == "idle":
            # Just show a simple feed
            cv2.putText(frame, "STATUS: IDLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Press 's' to start {SCAN_DURATION_SEC}s scan", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            display_frame = frame

        # ---
        # State: SCANNING (10-second window is active)
        # ---
        elif current_state == "scanning":
            time_elapsed = time.time() - scan_start_time
            time_remaining = SCAN_DURATION_SEC - time_elapsed

            # Run YOLO to show live boxes, but we don't act on these results
            results = model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            display_frame = results[0].plot()

            # Draw timer
            cv2.putText(display_frame, f"STATUS: SCANNING... {time_remaining:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, "Perform action now...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # --- Timer Finished ---
            if time_remaining <= 0:
                print("  SCAN COMPLETE. Capturing 'final' frame...")
                
                # Save the "after" shot!
                final_frame = frame.copy()
                
                # --- Run Analysis (using our imported logic) ---
                print("  Analyzing 'initial' frame (from 10s ago)...")
                init_results = model.predict(initial_frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
                init_detections = logic.parse_yolo_results(init_results[0], logic.CLASS_MAP)
                
                print("  Analyzing 'final' frame (from now)...")
                final_results = model.predict(final_frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
                final_detections = logic.parse_yolo_results(final_results[0], logic.CLASS_MAP)
                
                # Create config for the logic function
                config = {"image_width": frame_width}
                
                # Run the change analysis
                inventory_changes = logic.analyze_inventory_changes(
                    init_detections, 
                    final_detections, 
                    config, 
                    logic.CLASS_MAP
                )
                
                # Apply updates
                inventory = logic.apply_inventory_updates(inventory, inventory_changes)
                
                # Save to file
                logic.save_inventory(INVENTORY_FILE_PATH, inventory)
                
                print("  Analysis complete. Inventory saved.")
                print(f"  Final changes: {json.dumps(inventory_changes if inventory_changes else {})}")
                print("="*20 + "\n")
                
                # Reset state
                current_state = "idle"
                initial_frame = None
                scan_start_time = None

        # ---
        # Display / stream the final composed frame
        # ---
        # If MJPEG streaming is enabled, push JPEG frames into the buffer
        if '--mjpeg-port' in sys.argv:
            # encode to JPEG and update global
            try:
                ret, jpg = cv2.imencode('.jpg', display_frame)
                if ret:
                    with _latest_jpeg_lock:
                        # store raw bytes
                        global _latest_jpeg
                        _latest_jpeg = jpg.tobytes()
            except Exception:
                pass
        else:
            cv2.imshow("Action Scanner (Press 'q' to quit)", display_frame)

    # 5. Cleanup
    print("\n" + "="*40)
    print("Quitting... Final inventory saved.")
    print("FINAL INVENTORY:")
    print(json.dumps(inventory, indent=2))
    print("="*40)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Optional CLI: --mjpeg-port PORT
    mjpeg_server = None
    port = None
    if '--mjpeg-port' in sys.argv:
        try:
            idx = sys.argv.index('--mjpeg-port')
            port = int(sys.argv[idx + 1])
            mjpeg_server = _start_mjpeg_server(port)
        except Exception as e:
            print(f"Failed to start MJPEG server: {e}")

    try:
        main()
    finally:
        if mjpeg_server is not None:
            mjpeg_server.shutdown()