# road_damage_app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from pathlib import Path
import numpy as np

# --- Configuration ---
MODEL_WEIGHTS = 'best.pt'
IMG_SIZE = 512
CONF_DEFAULT = 0.25 # Default confidence threshold

# Use Streamlit's cache decorator to load the model only once.
# This prevents the model from reloading every time the user interacts with the app,
# which is crucial for performance.
@st.cache_resource
def load_yolo_model(model_path):
    """Loads the custom YOLOv8-GSAF model."""
    try:
        # The YOLO class automatically finds the custom modules from 'modules.py' 
        # because it is in the same directory.
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure 'best.pt' and 'modules.py' are in the same folder.")
        return None

def main():
    st.set_page_config(
        page_title="YOLOv8-GSAF Road Damage Detection", 
        layout="wide"
    )

    st.title("🛣️ YOLOv8-GSAF Road Damage Detection")
    st.subheader("Upload an image to detect D00, D10, D20, and D40 classes.")
    
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
        value=CONF_DEFAULT, 
        step=0.05
    )
    st.sidebar.markdown(f"**Model Parameters:**\n- Weights: `{MODEL_WEIGHTS}`\n- Image Size: `{IMG_SIZE}x{IMG_SIZE}`")
    
    # --- Main Interface ---
    uploaded_file = st.file_uploader(
        "Upload a road image...", 
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # Convert uploaded file to a PIL Image
        input_image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(input_image, caption='Uploaded Image', use_column_width=True)
        st.write("---")
        
        # Create columns for results
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Run Prediction", type="primary"):
                with st.spinner('Running YOLOv8-GSAF Detection...'):
                    # Convert PIL Image to a format model.predict() can handle (Numpy array)
                    img_array = np.array(input_image)

                    # Run inference
                    results = model.predict(
                        source=img_array, 
                        imgsz=IMG_SIZE, 
                        conf=confidence, 
                        verbose=False # Keep the console clean
                    )
                    
                    # Get the annotated image (with bounding boxes)
                    annotated_frame = results[0].plot()[:, :, ::-1] # BGR to RGB conversion for Streamlit

                    # Display the predicted image
                    with col2:
                        st.subheader("Prediction Result")
                        st.image(annotated_frame, caption='Detected Road Damages', use_column_width=True)

                    # Display raw results data
                    st.success("Detection Complete!")
                    
                    with st.expander("View Detection Details (JSON)"):
                        # Extract and show box/class data
                        boxes = results[0].boxes
                        if len(boxes) > 0:
                            st.dataframe({
                                'Class Name': [model.names[int(cls)] for cls in boxes.cls],
                                'Confidence': [f"{conf:.2f}" for conf in boxes.conf],
                                'BBox (Normalized)': [f"[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]" 
                                                      for x1, y1, x2, y2 in boxes.xyxyn.tolist()]
                            })
                        else:
                            st.info("No road damages detected in this image.")

if __name__ == "__main__":
    main()