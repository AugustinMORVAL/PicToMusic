import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from UI.statics import apply_custom_css, create_file_uploader, create_camera_input, display_tips

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

# Navigation buttons
col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    st.markdown("""
        <a href="app.py" style="text-decoration: none;">
            <button style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; background: white; cursor: pointer;">
                🏠 Home
            </button>
        </a>
    """, unsafe_allow_html=True)

with col_nav2:
    st.markdown("""
        <a href="pages/1_PParser.py" style="text-decoration: none;">
            <button style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; background: white; cursor: pointer;">
                🔄 Previous: PParser
            </button>
        </a>
    """, unsafe_allow_html=True)

with col_nav3:
    st.markdown("""
        <a href="pages/3_Final_Note_Detection.py" style="text-decoration: none;">
            <button style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; background: white; cursor: pointer;">
                🔄 Next: Final Note Detection
            </button>
        </a>
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Configuration")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
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
    
    with col2:
        st.markdown("### Post-Processing")
        
        min_note_distance = st.number_input(
            "Minimum Note Distance",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
            help="Minimum distance between detected notes"
        )
        
        staff_line_thickness = st.number_input(
            "Staff Line Thickness",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Thickness of staff lines for visualization"
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
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Staff Lines Detection")
                    st.image(results[0].plot(), caption="Staff Detection")
                
                with col2:
                    st.subheader("Detection Statistics")
                    st.markdown("### Detection Results")
                    st.markdown(f"- Notes detected: {len(results[0].boxes)}")
                    st.markdown(f"- Confidence: {results[0].boxes.conf.mean():.2f}")
                
                st.success("✨ YOLO detection completed!")
                
                st.markdown("""
                    <div class='info-box'>
                        Processing complete! Available options:
                        - ▶️ Play the converted music
                        - 💾 Download as MIDI file
                        - 🎼 View the musical notation
                        - 🔄 Continue to Final Note Detection
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔄 Continue to Final Note Detection"):
                    st.switch_page("pages/2_Final_Note_Detection.py")
                
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

display_tips() 