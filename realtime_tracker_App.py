# realtime_tracker_app.py
import streamlit as st
from ultralytics import YOLO
import numpy as np
import av 
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Configuration ---
MODEL_WEIGHTS = 'best.pt'
IMG_SIZE = 512
CONF_DEFAULT = 0.40 # Start with a higher default for tracking

# --- Model Loading ---
@st.cache_resource
def load_yolo_model(model_path):
    """Loads the custom YOLOv8-GSAF model."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure 'best.pt' and 'modules.py' are in the same folder.")
        return None

# --- NEW: The Real-Time Tracking Processor ---
class YoloVideoTransformer(VideoTransformerBase):
    def __init__(self, model, confidence, tracker_config):
        self.model = model
        self.confidence = confidence
        self.tracker_config = tracker_config

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        This callback method processes each frame.
        """
        img_bgr = frame.to_ndarray(format="bgr24")

        # --- KEY CHANGE: Use model.track() ---
        # We replace .predict() with .track()
        # 'persist=True' tells the tracker to remember IDs between frames
        # 'tracker=...' uses the config file we select (e.g., 'bytetrack.yaml')
        results = self.model.track(
            source=img_bgr,
            imgsz=IMG_SIZE,
            conf=self.confidence,
            persist=True, 
            tracker=self.tracker_config,
            verbose=False
        )
        
        # Get the annotated frame (with boxes AND track IDs)
        annotated_frame_bgr = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame_bgr, format="bgr24")

# --- Main App ---
def main():
    st.set_page_config(
        page_title="YOLOv8-GSAF Real-Time Tracking",
        layout="wide"
    )

    st.title("🛣️ YOLOv8-GSAF Real-Time Road Damage Tracker")
    st.subheader("Detects and tracks unique damage instances with an ID.")
    
    model = load_yolo_model(MODEL_WEIGHTS)
    if model is None:
        return
    
    # --- Sidebar Configuration ---
    st.sidebar.header("Model Settings")
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=CONF_DEFAULT,
        step=0.05
    )
    
    # --- NEW: Tracker Selection ---
    tracker_type = st.sidebar.selectbox(
        "Select Tracker",
        ("ByteTrack", "BoT-SORT")
    )
    tracker_config = "bytetrack.yaml" if tracker_type == "ByteTrack" else "botsort.yaml" #
    
    st.sidebar.markdown(f"**Model Parameters:**\n- Weights: `{MODEL_WEIGHTS}`\n- Image Size: `{IMG_SIZE}x{IMG_SIZE}`")
    st.sidebar.info(
        "A higher threshold (e.g., 0.40+) is recommended for real-time tracking "
        "to get stable IDs and reduce 'flickering'."
    )

    # --- Main Interface ---
    st.header("Live Webcam Feed")
    st.write("Click 'START' to begin your webcam feed and run tracking.")

    webrtc_streamer(
        key="yolo_stream",
        video_processor_factory=lambda: YoloVideoTransformer(
            model=model,
            confidence=confidence,
            tracker_config=tracker_config # Pass the selected tracker
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()