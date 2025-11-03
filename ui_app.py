import os
import time
import threading
from typing import Optional

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# Reuse existing logic
import inventory_logic as logic
from ultralytics import YOLO

# -----------------------------
# Configuration
# -----------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "my_model", "my_model.pt")
INVENTORY_FILE_PATH = os.path.join(APP_ROOT, "inventory.json")
CONFIDENCE_THRESHOLD = 0.5
SCAN_DURATION_SEC = 10

RTC_CONFIGURATION = RTCConfiguration(
	iceServers=[{"urls": ["stun:stun.l.google.com:19302"]}]
)

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
	return YOLO(path)


def ensure_session_state():
	ss = st.session_state
	if "is_scanning" not in ss:
		ss.is_scanning = False
	if "scan_start_ts" not in ss:
		ss.scan_start_ts = None
	if "initial_frame" not in ss:
		ss.initial_frame = None  # type: Optional[np.ndarray]
	if "final_frame" not in ss:
		ss.final_frame = None  # type: Optional[np.ndarray]
	if "last_changes" not in ss:
		ss.last_changes = {}
	if "inventory" not in ss:
		ss.inventory = logic.load_inventory(INVENTORY_FILE_PATH)
	if "processor_ref" not in ss:
		ss.processor_ref = None
	if "analyzing" not in ss:
		ss.analyzing = False


# -----------------------------
# Video Processor
# -----------------------------
class PreviewProcessor(VideoProcessorBase):
	def __init__(self):
		self.latest_frame: Optional[np.ndarray] = None
		self.overlay_text: Optional[str] = None

	def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
		img = frame.to_ndarray(format="bgr24")
		self.latest_frame = img

		# Draw overlay text (status/timer)
		if self.overlay_text:
			cv2.putText(
				img,
				self.overlay_text,
				(20, 40),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.0,
				(0, 0, 255) if "SCANNING" in self.overlay_text else (0, 255, 0),
				2,
			)

		return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------
# Analysis Thread
# -----------------------------

def run_analysis_async(initial_frame: np.ndarray, final_frame: np.ndarray):
	"""Run YOLO on initial and final frames, compute changes, update inventory & state."""
	ss = st.session_state
	try:
		model = load_model(MODEL_PATH)

		init_results = model.predict(initial_frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
		final_results = model.predict(final_frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

		init_dets = logic.parse_yolo_results(init_results[0], logic.CLASS_MAP)
		final_dets = logic.parse_yolo_results(final_results[0], logic.CLASS_MAP)

		config = {"image_width": initial_frame.shape[1]}
		changes = logic.analyze_inventory_changes(init_dets, final_dets, config, logic.CLASS_MAP)

		ss.inventory = logic.apply_inventory_updates(ss.inventory, changes)
		logic.save_inventory(INVENTORY_FILE_PATH, ss.inventory)
		ss.last_changes = changes
	finally:
		ss.analyzing = False
		ss.is_scanning = False
		ss.scan_start_ts = None
		ss.initial_frame = None
		ss.final_frame = None


# -----------------------------
# UI
# -----------------------------

def main():
	st.set_page_config(page_title="Action Scanner UI", layout="wide")
	ensure_session_state()
	ss = st.session_state

	st.title("Action Scanner")
	st.caption("Camera preview with a 10s scan timer, and live inventory updates.")

	col_preview, col_table = st.columns([3, 2])

	with col_preview:
		st.subheader("Camera Preview")
		webrtc_ctx = webrtc_streamer(
			key="scanner",
			mode=WebRtcMode.SENDRECV,
			video_processor_factory=PreviewProcessor,
			media_stream_constraints={"video": True, "audio": False},
			rtc_configuration=RTC_CONFIGURATION,
		)

		if webrtc_ctx and webrtc_ctx.video_processor:
			ss.processor_ref = webrtc_ctx.video_processor

		row1 = st.columns([1, 1, 2])
		with row1[0]:
			if st.button("Start 10s Scan", type="primary", disabled=ss.is_scanning or ss.processor_ref is None):
				# Capture initial frame
				if ss.processor_ref and ss.processor_ref.latest_frame is not None:
					ss.initial_frame = ss.processor_ref.latest_frame.copy()
					ss.is_scanning = True
					ss.scan_start_ts = time.time()
					ss.last_changes = {}
				else:
					st.warning("Camera not ready yet. Please wait a moment and try again.")

		with row1[1]:
			if st.button("Reset State", disabled=ss.is_scanning):
				ss.is_scanning = False
				ss.scan_start_ts = None
				ss.initial_frame = None
				ss.final_frame = None
				ss.last_changes = {}

		# Update overlay text on preview
		if ss.processor_ref is not None:
			if ss.is_scanning and ss.scan_start_ts is not None:
				elapsed = time.time() - ss.scan_start_ts
				remaining = max(0.0, SCAN_DURATION_SEC - elapsed)
				ss.processor_ref.overlay_text = f"SCANNING... {remaining:.1f}s"
			else:
				ss.processor_ref.overlay_text = "STATUS: IDLE"

		# When timer ends, capture final frame and run analysis (once)
		if ss.is_scanning and ss.scan_start_ts is not None and not ss.analyzing:
			elapsed = time.time() - ss.scan_start_ts
			if elapsed >= SCAN_DURATION_SEC:
				if ss.processor_ref and ss.processor_ref.latest_frame is not None:
					ss.final_frame = ss.processor_ref.latest_frame.copy()
					ss.analyzing = True
					# Launch analysis in background to avoid blocking UI
					threading.Thread(target=run_analysis_async, args=(ss.initial_frame, ss.final_frame), daemon=True).start()

	with col_table:
		st.subheader("Inventory")
		inv = ss.inventory or {}
		if inv:
			# Render as table
			st.dataframe(
				{
					"Item": list(inv.keys()),
					"Count": [inv[k] for k in inv.keys()],
				},
				use_container_width=True,
			)
		else:
			st.info("Inventory is empty.")

		if ss.last_changes:
			st.markdown("**Latest Update**")
			changes_view = {k: int(v) for k, v in ss.last_changes.items()}
			st.json(changes_view)

		st.caption(f"Inventory file: {INVENTORY_FILE_PATH}")

	# Auto-refresh the page while scanning or analyzing to keep timer and results live
	if ss.is_scanning or ss.analyzing:
		st.rerun()


if __name__ == "__main__":
	main()
