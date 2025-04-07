import streamlit as st
import numpy as np
import cv2
from UI.statics import apply_custom_css, create_file_uploader, create_camera_input, info_box
import pickle
from ultralytics import YOLO
from sonatabene.converter import yolo_to_abc, abc_to_midi, abc_to_audio, abc_to_musescore
from music21 import instrument
from io import BytesIO
from sonatabene.converter.converter_abc import INSTRUMENT_MAP
from midi2audio import FluidSynth
import tempfile
import os
from sonatabene.utils import get_musescore_path 
import webbrowser

st.set_page_config(
    page_title="Chopin - Note Classification",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="collapsed"
)


apply_custom_css()

st.title("🎼 Chopin - Note Classification")
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

    # Resize image while maintaining aspect ratio
    target_width = 640  
    height, width = image.shape[:2]
    scale = target_width / width
    new_height = int(height * scale)
    image = cv2.resize(image, (target_width, new_height))

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
            list(INSTRUMENT_MAP.keys()),
            help="Select the instrument for playback"
        )
    
    if st.button("🎵 Generate Music"):
        st.title("Music Generation Results...")
        with st.spinner("🎼 Generating your music..."):
            try:
                model = YOLO('models/chopin.pt')
                
                results = model.predict(
                    source=image,
                    conf=confidence_threshold,
                    iou=nms_threshold,
                    save=False,
                )
                
                st.subheader("Classified Notes")
                st.image(results[0].plot(), caption="Classified Note Detection")

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

                st.success("✨ Note classification completed!")
            
                st.subheader("Music Preview")

                abc_notation = yolo_to_abc(results)
                st.text("Generated ABC Notation:")
                st.code(abc_notation)
                
                try:
                    instrument_class = INSTRUMENT_MAP[instrument]
                    
                    midi_buffer = BytesIO()
                    abc_to_midi(abc_notation, midi_buffer, instrument=instrument_class, tempo_bpm=tempo)                    
                    midi_buffer.seek(0)
                    results_midi = midi_buffer.read()
                    with st.spinner("🎼 Converting MIDI to Audio..."):
                        if len(results_midi) > 0:
                            soundfont_path = os.path.abspath('.fluidsynth/default_sound_font.sf2')
                            fs = FluidSynth(sound_font=soundfont_path)                        
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as midi_file:
                                midi_file.write(results_midi)
                                midi_file_path = midi_file.name
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as audio_file:
                                wav_path = audio_file.name
                                fs.midi_to_audio(midi_file_path, wav_path)
                                with open(wav_path, 'rb') as wav_file:
                                    results_audio = wav_file.read()
                            
                            st.audio(wav_path, format='audio/wav')
                            st.success("✅ MIDI file generated and converted to audio successfully!")
                        else:
                            st.error("❌ Failed to generate MIDI file. The generated file is empty.")
                except Exception as e:
                    st.error(f"❌ Error generating MIDI file: {str(e)}")

                try:
                    musescore_path = get_musescore_path()

                    st.button("🎼 Open in MuseScore", 
                            on_click=lambda: abc_to_musescore(abc_notation, open=True, musescore_path=musescore_path), 
                            use_container_width=True)
                
                except (FileNotFoundError, OSError) as e:
                    st.info(f"{str(e)} [Click here to download MuseScore 4](https://musescore.org/en/download)")

                dl1, dl2, dl3 = st.columns(3)

                with dl1:
                    results_pickle = pickle.dumps(results)
                    
                    st.download_button(
                        label="💾 Download YOLO Classification Data",
                        data=results_pickle,
                        file_name=f"{st.session_state['file_name']}_yolo_classification.pkl",
                        mime="application/octet-stream",
                        help="Download the YOLO classification results in pickle format",
                        use_container_width=True
                    )
                
                with dl2:
                    st.download_button(
                        label="🎵 Download MIDI File",
                        data=results_midi,
                        file_name=f"{st.session_state['file_name']}.mid",
                        mime="audio/midi",
                        help="Download the MIDI file",
                        use_container_width=True
                    )
                
                with dl3:
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
