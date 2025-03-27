import streamlit as st
import time
import numpy as np 
from UI.streamlit_app_logic import parse_music_sheet
import cv2

st.set_page_config(
    page_title="Pic to Music App",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS 
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .upload-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-message {
        padding: 1rem;
        background-color: #dff0d8;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stExpander {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stTab {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎼 Pic to Music App")
st.markdown("""
    <div class='info-box'>
        Transform your music sheets into playable music! Simply upload an image or take a photo and we'll do the rest. 🎵
    </div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 Upload File", "📸 Take Photo"])

# Tab 1: Upload File
with tab1:
    st.markdown("### 📤 Upload Your Music Sheet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            try:
                file_name = uploaded_file.name
                st.markdown(f"""
                    <div class='success-message'>
                        ✅ File "{file_name}" uploaded successfully!
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with col2:
        if uploaded_file is not None:
            with st.spinner("Processing image..."):
                st.image(uploaded_file, caption="Preview")

# Tab 2: Take Photo
with tab2:
    st.markdown("### 📸 Capture Music Sheet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        camera_input = st.camera_input(
            "Take a picture",
            help="Make sure the sheet music is well-lit and clearly visible"
        )
        
        if camera_input is not None:
            try:
                st.markdown("""
                    <div class='success-message'>
                        ✅ Image captured successfully!
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with col2:
        if camera_input is not None:
            with st.spinner("Processing image..."):
                st.image(camera_input, caption="Preview")

if camera_input is not None or uploaded_file is not None:
    st.markdown("---")
    
    st.title("🔧 Image Processing Parameters")
    
    # resize_max_dim = st.slider("Image Resolution", 800, 2000, 1600, 100, 
    #                           help="Maximum dimension for image resizing. Higher values provide more detail but slower processing.")
    
    col_staff, col_notes = st.columns(2)
    
    with col_staff:
        st.markdown("### Staff Line Detection")
        
        staff_dilate_iterations = st.number_input("Staff Line Dilation", 
                                                min_value=1, max_value=10, value=3, step=1,
                                                help="Number of dilation iterations for staff lines detection")
        
        staff_min_contour_area = st.number_input("Min Staff Contour Area", 
                                               min_value=100, max_value=20000, value=10000, step=1000,
                                               help="Minimum contour area for staff detection")
        
        staff_pad_size = st.number_input("Staff Padding", 
                                       min_value=0, max_value=75, value=0, step=5,
                                       help="Adding padding around image to avoid edge effects")
    
    # Note detection parameters
    with col_notes:
        st.markdown("### Note Detection")
        
        note_dilate_iterations = st.number_input("Note Dilation", 
                                               min_value=1, max_value=10, value=3, step=1,
                                               help="Number of dilation iterations for note detection")
        
        note_min_contour_area = st.number_input("Min Note Contour Area", 
                                              min_value=10, max_value=1000, value=100, step=25,
                                              help="Minimum contour area for note detection")
        
        # note_pad_size = st.number_input("Note Padding", 
        #                               min_value=0, max_value=30, value=0, step=2,
        #                               help="Adding padding around staff lines to avoid edge effects")
        
        max_horizontal_distance = st.number_input("Max Horizontal Distance", 
                                                min_value=0, max_value=10, value=2, step=1,
                                                help="Maximum horizontal distance between notes")
        
    overlap_threshold = st.slider("Overlap Threshold", 
                                    min_value=0.1, max_value=0.9, value=0.5, step=0.1,
                                    help="Threshold for determining overlapping elements and merge them")
    
    if st.button("🎵 Parse Music Sheet"):
        st.title("Parsing results...")
        with st.spinner("🎼 Converting your sheet music..."):
            try:
                
                image_source = uploaded_file if uploaded_file is not None else camera_input
                
                params = {
                    # 'resize_max_dim': int(resize_max_dim),
                    'staff_dilate_iterations': int(staff_dilate_iterations),
                    'staff_min_contour_area': int(staff_min_contour_area),
                    'staff_pad_size': int(staff_pad_size),
                    'note_dilate_iterations': int(note_dilate_iterations),
                    'note_min_contour_area': int(note_min_contour_area),
                    'note_pad_size': 0,
                    'max_horizontal_distance': int(max_horizontal_distance),
                    'overlap_threshold': float(overlap_threshold)
                }

                # Convert to numpy array
                image_bytes = image_source.getvalue()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                progress_bar = st.progress(0)
                
                staff_lines, staff_visualization, notes_visualization, all_notes = parse_music_sheet(image, progress_bar, params)
                
                col_staff_analysis, col_notes_analysis = st.columns(2)
                
                # Column 1: Staff Visualization
                with col_staff_analysis:
                    st.subheader("Staff Lines Analysis")
                    st.image(staff_visualization, caption="Staff Lines and Notes Detection")
                    
                    st.markdown("""
                        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                            <div style='display: flex; justify-content: space-around; align-items: center;'>
                                <div style='display: flex; align-items: center;'>
                                    <div style='width: 30px; height: 3px; background-color: #00FF00; margin-right: 5px;'></div>
                                    <span>Staff contours</span>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Column 2: Note detection results
                with col_notes_analysis:
                    st.subheader("Notes Detection Analysis")
                    st.image(notes_visualization, caption="Notes Detection")
                    
                    st.markdown("""
                        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                            <div style='display: flex; justify-content: space-around; align-items: center;'>
                                <div style='display: flex; align-items: center;'>
                                    <div style='width: 30px; height: 3px; background-color: #FF0000; margin-right: 5px;'></div>
                                    <span>Note contours</span>
                                </div>
                                <div style='display: flex; align-items: center;'>
                                    <div style='width: 30px; height: 3px; background-color: #0000FF; margin-right: 5px;'></div>
                                    <span>Note boundaries</span>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")   
                _, col_metrics1, col_metrics2, col_metrics3, _= st.columns(5)
                
                with col_metrics1:
                    st.metric(
                        label="Staff Lines Detected", 
                        value=len(staff_lines)
                    )
                
                with col_metrics2:
                    total_notes = sum(len(staff_notes) for staff_notes in all_notes)
                    st.metric(
                        label="Total Notes Detected", 
                        value=total_notes
                    )
                
                with col_metrics3:
                    avg_notes_per_staff = round(total_notes / len(staff_lines), 1)
                    st.metric(
                        label="Avg. Notes per Staff", 
                        value=avg_notes_per_staff
                    )
                
                st.success("✨ Music sheet successfully parsed!")
        
                st.markdown("""
                    <div class='info-box'>
                        Processing complete! Available options:
                        - ▶️ Play the converted music
                        - 💾 Download as MIDI file
                        - 🎼 View the musical notation
                    </div>
                """, unsafe_allow_html=True)
                
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
        if st.button("🎵 Convert to Music"):
            st.subheader("Converting to music...")
            with st.spinner("🎼 Converting to music..."):
                pass
        
with st.expander("ℹ️ Tips for Best Results"):
    st.markdown("""
        ### 📝 Guidelines for Best Results
        
        #### When Uploading Files:
        - Use high-resolution images
        - Ensure the sheet music is well-lit
        - Avoid glare or shadows on the page
        - Make sure the entire sheet is visible
        
        #### When Taking Photos:
        - Hold your device steady
        - Use good lighting
        - Avoid shadows
        - Center the sheet music in frame
        - Keep the camera parallel to the page
        
        ### 🎵 Supported Features
        - Standard musical notation
        - Multiple staves
        - Various time signatures
        - Different key signatures
    """)