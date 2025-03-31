import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from UI.statics import apply_custom_css, create_file_uploader, create_camera_input, display_tips
import pickle

st.set_page_config(
    page_title="Pic to Music App - YOLO Parser",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_custom_css()

st.title("🎼 Pic to Music App - YOLO Parser")
st.markdown("""
    <div class='info-box'>
        Transform your music sheets into playable music using our YOLO-based detection method! This method uses advanced deep learning to detect musical elements. 🎵
    </div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 Upload File", "📸 Take Photo"])

# Tab 1: Upload File
with tab1:
    uploaded_file = create_file_uploader()

# Tab 2: Take Photo
with tab2:
    camera_input = create_camera_input()

if camera_input is not None or uploaded_file is not None:
    if camera_input is not None:
        image = cv2.imdecode(np.frombuffer(camera_input.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    else:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    st.markdown("---")
    
    st.title("🔧 YOLO Model Parameters")
    

    st.markdown("### Model Configuration")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    nms_threshold = st.slider(
        "NMS Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Non-Maximum Suppression threshold"
    )
    
    if st.button("🎵 Detect Musical Elements"):
        st.title("Detection results...")
        with st.spinner("🎼 Processing your sheet music with YOLO..."):
            try:
                model = YOLO('models/yparser.pt')
                
                results = model.predict(
                    source=image,
                    conf=confidence_threshold,
                    iou=nms_threshold,
                    save=False,
                    project='resources/output/YOLO/tests',
                )
                
                
                st.subheader("Musical Elements Detection")
                st.image(results[0].plot(), caption="Musical Elements Detection")

                st.markdown("---")   
                _, col_metrics1, col_metrics2, col_metrics3, _ = st.columns(5)
                
                with col_metrics1:
                    st.metric(
                        label="Total Objects Detected", 
                        value=len(results[0].boxes)
                    )
                
                with col_metrics2:
                    st.metric(
                        label="Average Confidence", 
                        value=f"{results[0].boxes.conf.mean():.2f}"
                    )
                
                with col_metrics3:
                    st.metric(
                        label="Detected Classes", 
                        value=len(set(results[0].boxes.cls.tolist()))
                    )
                
                st.success("✨ YOLO detection completed!")
                
                results_bytes = pickle.dumps(results[0])
                
                st.download_button(
                    label="💾 Download YOLO Detection Data",
                    data=results_bytes,
                    file_name=f"{st.session_state['file_name']}_yolo_detection.pkl",
                    mime="application/octet-stream",
                    help="Download the YOLO detection results in pickle format",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error processing sheet music: {str(e)}")
                st.markdown("""
                    <div class='info-box' style='background-color: #ffebee;'>
                        Tips for better results:
                        - Ensure the image is clear and well-lit
                        - Make sure the sheet music is properly aligned
                        - Try adjusting the image contrast
                    </div>
                """, unsafe_allow_html=True)