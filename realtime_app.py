# realtime_app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from pathlib import Path
import numpy as np
import av # New import for video processing
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase # New imports

# --- Configuration ---
MODEL_WEIGHTS = 'best.pt'
IMG_SIZE = 512
CONF_DEFAULT = 0.25 # Default confidence threshold

# --- Model Loading ---
# This function is unchanged and still benefits from caching
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

# --- NEW: The Real-Time Video Processor ---
# This class is the core of the real-time functionality.
# It receives video frames, runs your model, and returns the annotated frames.
class YoloVideoTransformer(VideoTransformerBase):
    def __init__(self, model, confidence):
        self.model = model
        self.confidence = confidence

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        This is the callback method that processes each frame.
        """
        # 1. Convert the frame to a NumPy array
        # The frame format from streamlit-webrtc is BGR
        img_bgr = frame.to_ndarray(format="bgr24")

        # 2. Run your YOLOv8-GSAF model
        # The model.predict() can take BGR numpy arrays directly
        results = self.model.predict(
            source=img_bgr,
            imgsz=IMG_SIZE,
            conf=self.confidence,
            verbose=False # Keep the console clean
        )
        
        # 3. Get the annotated frame (with boxes)
        # results[0].plot() returns a BGR numpy array
        annotated_frame_bgr = results[0].plot()

        # 4. Return the new annotated frame
        return av.VideoFrame.from_ndarray(annotated_frame_bgr, format="bgr24")

# --- Main App ---
def main():
    st.set_page_config(
        page_title="YOLOv8-GSAF Real-Time Detection",
        layout="wide"
    )

    st.title("🛣️ YOLOv8-GSAF Real-Time Road Damage Detection")
    st.subheader("Using your webcam or a pre-recorded video file.")
    
    # 1. Load the Model
    model = load_yolo_model(MODEL_WEIGHTS)
    if model is None:
        return
    
    # --- Sidebar Configuration ---
    st.sidebar.header("Model Settings")
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=CONF_DEFAULT, # Start with your data-backed 0.25
        step=0.05
    )
    st.sidebar.markdown(f"**Model Parameters:**\n- Weights: `{MODEL_WEIGHTS}`\n- Image Size: `{IMG_SIZE}x{IMG_SIZE}`")
    
    st.sidebar.info(
        "For live recording, start with a higher threshold (e.g., 0.40) "
        "to prevent 'flickering' boxes and reduce false positives."
    )

    # --- Main Interface ---
    st.header("Live Webcam Feed")
    st.write("Click 'START' to begin your webcam feed and run detection.")

    # 5. Run the Real-Time Streamer!
    # This is the key component.
    webrtc_streamer(
        key="yolo_stream",
        # Pass an instance of our processor class
        video_processor_factory=lambda: YoloVideoTransformer(
            model=model,
            confidence=confidence
        ),
        # Configure the video stream (e.g., to use a front/back camera)
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()