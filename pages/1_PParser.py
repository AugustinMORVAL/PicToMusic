import streamlit as st
import numpy as np
from UI.pparser_app_logic import parse_music_sheet
from UI.statics import apply_custom_css, create_file_uploader, create_camera_input, display_tips
from p2m.parser import PParser
import cv2
import pickle

st.set_page_config(
    page_title="Pic to Music App - PParser",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_custom_css()

st.title("Pic to Music App - PParser")
st.markdown("""
    <div class='info-box'>
        Transform your music sheets into playable music using our Parser method! Simply upload an image or take a photo and we'll do the rest. 🎵
    </div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 Upload File", "📸 Take Photo"])

with tab1:
    uploaded_file = create_file_uploader()

with tab2:
    camera_input = create_camera_input()

if camera_input is not None or uploaded_file is not None:
    st.markdown("---")
    
    st.title("🔧 Image Processing Parameters")
    
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
    
    with col_notes:
        st.markdown("### Note Detection")
        
        note_dilate_iterations = st.number_input("Note Dilation", 
                                               min_value=1, max_value=10, value=3, step=1,
                                               help="Number of dilation iterations for note detection")
        
        note_min_contour_area = st.number_input("Min Note Contour Area", 
                                              min_value=10, max_value=1000, value=100, step=25,
                                              help="Minimum contour area for note detection")
        
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
                    'staff_dilate_iterations': int(staff_dilate_iterations),
                    'staff_min_contour_area': int(staff_min_contour_area),
                    'staff_pad_size': int(staff_pad_size),
                    'note_dilate_iterations': int(note_dilate_iterations),
                    'note_min_contour_area': int(note_min_contour_area),
                    'note_pad_size': 0,
                    'max_horizontal_distance': int(max_horizontal_distance),
                    'overlap_threshold': float(overlap_threshold)
                }

                image_bytes = image_source.getvalue()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                progress_bar = st.progress(0)
                
                staff_lines, staff_visualization, notes_visualization, all_notes = parse_music_sheet(image, progress_bar, params)
                
                col_staff_analysis, col_notes_analysis = st.columns(2)
                
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

                staff_lines_bytes = pickle.dumps(staff_lines)
                
                st.download_button(
                    label="💾 Download Staff Lines Data",
                    data=staff_lines_bytes,
                    file_name=f"{st.session_state['file_name']}_staff_lines.pkl",
                    mime="application/octet-stream",
                    help="Download the detected staff lines data in pickle format", 
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