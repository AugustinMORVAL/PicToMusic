import streamlit as st
import numpy as np
import cv2
from UI.statics import apply_custom_css, create_file_uploader, create_camera_input, display_tips

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
        <a href="pages/2_YOLO_Parser.py" style="text-decoration: none;">
            <button style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; background: white; cursor: pointer;">
                🔄 Previous: YOLO Parser
            </button>
        </a>
    """, unsafe_allow_html=True)

with col_nav3:
    st.markdown("""
        <a href="pages/1_PParser.py" style="text-decoration: none;">
            <button style="width: 100%; padding: 0.5rem; border: 1px solid #ddd; background: white; cursor: pointer;">
                🔄 Previous: PParser
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
    
    st.title("🔧 Note Refinement Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Note Refinement")
        
        note_confidence = st.slider(
            "Note Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence for note detection"
        )
        
        note_merge_distance = st.number_input(
            "Note Merge Distance",
            min_value=0,
            max_value=50,
            value=10,
            step=5,
            help="Distance threshold for merging overlapping notes"
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
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Refined Notes")
                    st.image(image, caption="Refined Note Detection")
                
                with col2:
                    st.subheader("Music Preview")
                    st.markdown("""
                        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;'>
                            <h3>🎵 Audio Player (Coming Soon)</h3>
                            <p>Your generated music will appear here</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.success("✨ Music generation completed!")
                
                st.markdown("""
                    <div class='info-box'>
                        Processing complete! Available options:
                        - ▶️ Play the generated music
                        - 💾 Download as MIDI file
                        - 🎼 View the musical notation
                        - 📊 View note statistics
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating music: {str(e)}")
                st.markdown("""
                    <div class='info-box' style='background-color: #ffebee;'>
                        Tips for better results:
                        - Ensure the note detection is accurate
                        - Adjust the tempo if needed
                        - Try different instruments for playback
                    </div>
                """, unsafe_allow_html=True)

display_tips() 