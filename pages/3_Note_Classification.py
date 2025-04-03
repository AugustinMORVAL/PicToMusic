import streamlit as st
import numpy as np
import cv2
from UI.statics import apply_custom_css, create_file_uploader, create_camera_input, info_box
import pickle
st.set_page_config(
    page_title="Pic to Music App - Note Detection",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="collapsed"
)


apply_custom_css()

st.title("🎼 Pic to Music App - Final Note Detection")
st.markdown("""
    <div class='info-box'>
        Finalize your music sheet processing and generate playable music! This page allows you to refine the detected notes and convert them into MIDI files. 🎵
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
    
    st.title("🔧 Note Refinement Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Note Refinement")
        
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
        st.markdown("### Music Generation")
        
        tempo = st.number_input(
            "Tempo (BPM)",
            min_value=40,
            max_value=200,
            value=120,
            step=5,
            help="Beats per minute for the generated music"
        )
        
        instrument = st.selectbox(
            "Instrument",
            ["Piano", "Guitar", "Violin", "Flute", "Trumpet"],
            help="Select the instrument for playback"
        )
    
    if st.button("🎵 Generate Music"):
        st.title("Music Generation Results...")
        with st.spinner("🎼 Generating your music..."):
            try:
                # Process the image directly
                # TODO: Implement note detection here
                results = 'results'
                
                
                st.subheader("Classified Notes")
                st.image(image, caption="Classified Note Detection")
            
                st.subheader("Music Preview")

                # TODO: Implement audio generation here
                # TODO: Implement audio player here

                st.markdown("""
                    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 30px;'>
                        <h3>🎵 Audio Player (Coming Soon)</h3>
                        <p>Your generated music will appear here</p>
                    </div>
                """, unsafe_allow_html=True)

                # st.markdown("---")   
                # _, col_metrics1, col_metrics2, col_metrics3, _ = st.columns(5)
                
                # with col_metrics1:
                #     st.metric(
                #         label="Total Objects Detected", 
                #         value=len(results[0].boxes)
                #     )
                
                # with col_metrics2:
                #     st.metric(
                #         label="Average Confidence", 
                #         value=f"{results[0].boxes.conf.mean():.2f}"
                #     )
                
                # with col_metrics3:
                #     st.metric(
                #         label="Detected Classes", 
                #         value=len(set(results[0].boxes.cls.tolist()))
                #     )
                
                st.success("✨ Note classification completed!")

                dl1, dl2 = st.columns(3)

                with dl1:
                    results_pickle = 'TO DO'
                    
                    st.download_button(
                        label="💾 Download YOLO Classification Data",
                        data=results_pickle,
                        file_name=f"{st.session_state['file_name']}_yolo_classification.pkl",
                        mime="application/octet-stream",
                        help="Download the YOLO classification results in pickle format",
                        use_container_width=True
                    )
                
                with dl2:
                    results_audio = 'TO DO'
                    st.download_button(
                        label="🎵 Download Audio File",
                        data=results_audio,
                        file_name=f"{st.session_state['file_name']}.wav",
                        mime="audio/wav",
                        help="Download the audio file",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Error classifying notes: {str(e)}")
                st.markdown("""
                    <div class='info-box' style='background-color: #ffebee;'>
                        Tips for better results:
                        - Ensure the note detection is accurate
                        - Adjust the tempo if needed
                        - Try different instruments for playback
                    </div>
                """, unsafe_allow_html=True)
