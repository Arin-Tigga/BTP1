import streamlit as st
import time
from pathlib import Path

st.set_page_config(page_title="Action Scanner (Webcam)", layout="wide")

st.title("Action Scanner — Webcam")

try:
    import inventory_logic as logic
except Exception as e:
    st.error(f"Failed to import inventory logic: {e}")
    raise

try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
except Exception:
    st.error("streamlit-webrtc not installed. Install with 'pip install streamlit-webrtc'")
    raise

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

inv_path = st.sidebar.text_input("Inventory JSON path", value="inventory.json")
model_path = st.sidebar.text_input("Model path (optional)", value="my_model/my_model.pt")
confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)

inv_file = Path(inv_path)
inventory = logic.load_inventory(str(inv_file)) if inv_file.exists() else {}


class FrameCaptureTransformer(VideoTransformerBase):
    def __init__(self):
        self.initial = None
        self.final = None
        self.latest_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # store latest frame on the transformer instance so the main thread can read it
        # (writing to st.session_state from the transformer worker is unreliable)
        self.latest_frame = img
        # optionally draw boxes when last visuals exist
        if 'last_initial_vis' in st.session_state:
            pass
        return img


webrtc_ctx = webrtc_streamer(key="scanner", video_transformer_factory=FrameCaptureTransformer, media_stream_constraints={"video": True, "audio": False})

col1, col2 = st.columns([2, 1])

with col2:
    st.header("Controls & Inventory")

    # Timed scan flow (Start Scan -> capture initial -> wait SCAN_DURATION -> capture final -> analyze)
    SCAN_DURATION = st.sidebar.number_input("Scan duration (s)", min_value=1, max_value=60, value=10)

    if 'scan_state' not in st.session_state:
        st.session_state['scan_state'] = 'idle'  # idle, counting, analyzing
    if 'scan_start_time' not in st.session_state:
        st.session_state['scan_start_time'] = None

    if st.button("Start Scan") and st.session_state['scan_state'] == 'idle':
        # get latest frame from the webrtc transformer if available
        latest = None
        if webrtc_ctx.video_transformer is not None:
            latest = getattr(webrtc_ctx.video_transformer, 'latest_frame', None)
        # fallback to session_state (older code path)
        if latest is None and 'latest_frame' in st.session_state:
            latest = st.session_state['latest_frame']

        if latest is None:
            st.warning("No live frame available to capture initial")
        else:
            st.session_state['initial_frame'] = latest.copy()
            st.session_state['scan_start_time'] = time.time()
            st.session_state['scan_state'] = 'counting'
            st.rerun()

    if st.session_state['scan_state'] == 'counting':
        elapsed = time.time() - st.session_state['scan_start_time']
        remaining = SCAN_DURATION - elapsed
        if remaining > 0:
            st.info(f"Scanning... {remaining:.1f}s remaining")
            st.rerun()
        else:
            # time's up -> capture final and analyze
            # obtain latest frame same as above
            latest = None
            if webrtc_ctx.video_transformer is not None:
                latest = getattr(webrtc_ctx.video_transformer, 'latest_frame', None)
            if latest is None and 'latest_frame' in st.session_state:
                latest = st.session_state['latest_frame']

            if latest is None:
                st.error("Failed to capture final frame")
                st.session_state['scan_state'] = 'idle'
            else:
                st.session_state['final_frame'] = latest.copy()
                st.session_state['scan_state'] = 'analyzing'

    if st.session_state['scan_state'] == 'analyzing':
        st.info("Running analysis...")
        try:
            init = st.session_state.get('initial_frame')
            final = st.session_state.get('final_frame')
            if init is None or final is None:
                raise RuntimeError("Initial or final frame missing")

            if YOLO_AVAILABLE:
                model = YOLO(model_path)
                init_res = model.predict(init, verbose=False, conf=confidence)
                final_res = model.predict(final, verbose=False, conf=confidence)
                init_dets = logic.parse_yolo_results(init_res[0], logic.CLASS_MAP)
                final_dets = logic.parse_yolo_results(final_res[0], logic.CLASS_MAP)
            else:
                init_dets = []
                final_dets = []

            cfg = {"image_width": final.shape[1]}
            changes = logic.analyze_inventory_changes(init_dets, final_dets, cfg, logic.CLASS_MAP)
            updated = logic.apply_inventory_updates(inventory, changes)
            logic.save_inventory(str(inv_file), updated)
            st.session_state['last_changes'] = changes
            st.success("Analysis complete — inventory updated")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
        finally:
            st.session_state['scan_state'] = 'idle'

    st.subheader("Inventory")
    st.write(inventory)

