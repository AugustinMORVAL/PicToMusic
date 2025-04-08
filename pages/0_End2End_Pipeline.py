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
if "image_uploaded" not in st.session_state:
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

    if uploaded_file:
        st.session_state["image_uploaded"] = uploaded_file
    elif camera_input:
        st.session_state["image_uploaded"] = camera_input

    if st.button("🔄 Upload another image"):
        del st.session_state["image_uploaded"]
        st.rerun()

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
        parser.load_image(st.session_state["image_uploaded"])
        staves = parser.find_staff_lines(
            dilation_iterations=staff_line_dilation,  # <- usa el nombre real
            min_contour_area=min_staff_area,                  # <- usa el nombre real
        )
        st.session_state["staves"] = staves
        st.image([s.image for s in staves], caption="Lignes de portée détectées")

    # 2 - DÉTECTION DE NOTES AVEC CHOPIN
    if "staves" in st.session_state and st.button("2️⃣ Détection des notes avec le modèle Chopin"):
        first_staff_img = st.session_state["staves"][0].image
        model_chopin = YOLO("models/chopin.pt")
        results_chopin = model_chopin.predict(first_staff_img, conf=0.55, iou=0.5)[0]
        st.session_state["results_chopin"] = results_chopin
        st.image(results_chopin.plot(), caption="Notes classifiées")

    # 3 - TRADUCTION EN ABC
    if "results_chopin" in st.session_state and st.button("3️⃣ Traduction YOLO -> ABC Notation"):
        abc_code = yolo_to_abc([st.session_state["results_chopin"]])
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
