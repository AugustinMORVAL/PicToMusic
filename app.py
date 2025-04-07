import streamlit as st
from UI.statics import apply_custom_css

st.set_page_config(
    page_title="Sonat'App",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

apply_custom_css()

st.markdown("""
    <div class='hero-section'>
        <h1 style='font-size: 3rem; margin-bottom: 1rem;'>🎼 Sonat'App</h1>
        <p style='font-size: 1.2rem;'>Transform your sheet music into playable music using advanced AI and computer vision techniques</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    ### About the Project
    SonataBene is an innovative library that converts sheet music images into playable music. 
    Our solution combines multiple advanced techniques to provide accurate and reliable music transcription:
""")

# Feature Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class='feature-card'>
            <h3>🎯 PParser</h3>
            <p>Computer vision approach for staffline and note detection using advanced image processing techniques</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class='feature-card'>
            <h3>🤖 Bach Detection</h3>
            <p>Deep learning-based detection of musical elements using state-of-the-art object detection models</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class='feature-card'>
            <h3>🎵 Chopin Classification</h3>
            <p>Deep learning-based classification of musical elements using state-of-the-art object classification models</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class='feature-card'>
            <h3>🎹 Music Generation</h3>
            <p>Convert detected notes into ABC notation and generate playable MIDI files</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("### How would you like to proceed?")

col_demo, col_pages = st.columns(2)

with col_demo:
    st.markdown("""
        <div class='demo-section'>
            <div>
                <h3>🎯 Guided Demo</h3>
                <p>Follow our step-by-step pipeline to process your sheet music</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("▶️ Start Guided Demo", key="guided_demo", use_container_width=True):
        st.switch_page("pages/1_PParser.py")

with col_pages:
    st.markdown("""
        <div class='demo-section'>
            <div>
                <h3>🔍 Individual Pages</h3>
                <p>Access specific functionality directly</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    

    if st.button("📝 PParser", key="pparser_btn", help="Staff line detection tool", use_container_width=True):
        st.switch_page("pages/1_PParser.py")
    if st.button("🔍 YOLO Parser", key="yolo_btn", help="Note detection using YOLO", use_container_width=True):
        st.switch_page("pages/2_YOLO_Parser.py")
    if st.button("🎵 YOLO Classification", key="classification_btn", help="Classify detected notes", use_container_width=True):
        st.switch_page("pages/3_Note_Classification.py")

st.markdown("### Processing Pipeline")
st.markdown("""
    <div class='pipeline-section'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
            <div class='pipeline-step' style='text-align: center; flex: 1;'>
                <h4>1. Score Parsing</h4>
                <p>Staff line detection</p>
            </div>
            <div style='flex: 1; text-align: center; color: #666;'>→</div>
            <div class='pipeline-step' style='text-align: center; flex: 1;'>
                <h4>2. Note Detection</h4>
                <p>Note detection</p>
            </div>
            <div style='flex: 1; text-align: center; color: #666;'>→</div>
            <div class='pipeline-step' style='text-align: center; flex: 1;'>
                <h4>3. Note Classification</h4>
                <p>Note classification</p>
            </div>
            <div style='flex: 1; text-align: center; color: #666;'>→</div>
            <div class='pipeline-step' style='text-align: center; flex: 1;'>
                <h4>4. Music Generation</h4>
                <p>ABC format & MIDI</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)