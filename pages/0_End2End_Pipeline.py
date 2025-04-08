import streamlit as st
st.set_page_config(page_title="Full Pipeline", page_icon="🎵", layout="wide")

import numpy as np
import cv2
from io import BytesIO
import tempfile
import os
from UI.statics import apply_custom_css, create_file_uploader, create_camera_input
from sonatabene.model import predict
from ultralytics import YOLO
from sonatabene.converter import yolo_to_abc, abc_to_midi
from sonatabene.converter.converter_abc import INSTRUMENT_MAP
from midi2audio import FluidSynth
from sonatabene.parser import PParser

apply_custom_css()
st.title("🎼 End-to-End Pipeline : From Sheet to Sound")

# IMAGE LOADING
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

    # PARAMÈTRES DE DÉTECTION
    st.markdown("### 🛠️ Paramètres de traitement")
    col1, col2 = st.columns(2)
    with col1:
        staff_line_dilation = st.slider("Staff Line Dilation", 1, 10, 3)
    with col2:
        min_staff_area = st.slider("Min Staff Contour Area", 1000, 20000, 10000)

    # 1 - DÉCOUPAGE STAFFLINES
    if st.button("1️⃣ Découpage des lignes de portée avec PParser"):
        parser = PParser()
        parser.load_image(image)
        staves = parser.find_staff_lines(
            dilate_iterations=staff_line_dilation,  # <- usa el nombre real
            min_contour_area=min_staff_area,                  # <- usa el nombre real
        )
        st.session_state["staves"] = staves
        for i, staff in enumerate(staves):
            st.image(staff.image, caption=f"staff {i+1}")
    

    staves = [cv2.cvtColor(staff.image, cv2.COLOR_RGB2BGR) for staff in st.session_state["staves"]]

    # 2 - DÉTECTION DE NOTES AVEC CHOPIN
    if "staves" in st.session_state and st.button("2️⃣ Détection des notes avec le modèle Chopin"):
        st.session_state["predictions"] = []
        for i, staff in enumerate(staves):
            result = predict(image=staff, model_path="models/chopin.pt")[0]
            st.session_state["predictions"].append(result)
            st.image(result.plot(), caption=f"Notes détectées sur le staff {i+1}")


    # 3 - TRADUCTION EN ABC
    if "predictions" in st.session_state and st.button("3️⃣ Traduction YOLO -> ABC Notation"):

        abc_code = yolo_to_abc(st.session_state["predictions"])
        st.session_state["abc_code"] = abc_code
        st.code(abc_code, language="abc")

    # 4 - GÉNÉRATION AUDIO
    if "abc_code" in st.session_state:
        st.markdown("### 🎹 Génération Audio")
        instrument_name = st.selectbox("Instrument", list(INSTRUMENT_MAP.keys()))
        tempo = st.slider("Tempo", 40, 200, 120)

        if st.button("4️⃣ Générer & Écouter"):
            try:
                instrument_class = INSTRUMENT_MAP[instrument_name]
                midi_buffer = BytesIO()
                abc_to_midi(st.session_state["abc_code"], midi_buffer, instrument=instrument_class, tempo_bpm=tempo)
                midi_buffer.seek(0)
                midi_data = midi_buffer.read()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as midif:
                    midif.write(midi_data)
                    midi_path = midif.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wavf:
                    wav_path = wavf.name
                    soundfont_path = os.path.abspath(".fluidsynth/default_sound_font.sf2")
                    fs = FluidSynth(sound_font=soundfont_path)
                    fs.midi_to_audio(midi_path, wav_path)

                st.audio(wav_path, format="audio/wav")
                st.success("✅ Audio généré avec succès !")

            except FileNotFoundError:
                st.error("❌ Fluidsynth non trouvé. Veuillez l'installer.")
