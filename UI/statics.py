import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
    /* Main layout and common elements */
    .main {
        padding: 2rem;
    }

    /* Tabs and UI Elements */
    .stTabs {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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

    /* Buttons */
    .stButton>button {
        width: 100%;
        padding: 0.8rem;
        background-color: white;
        border: 1px solid #ddd;
        cursor: pointer;
        text-align: center;
        color: black;
        display: inline-block;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        border-radius: 8px;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        background-color: #f0f0f0;
        border-color: #2E86C1;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Headers */
    h1 {
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
    }

    /* Sections */
    .hero-section {
        padding: 2rem;
        border-radius: 15px;
        color: black;
        text-align: center;
        border: 1px solid #ddd;
        margin-bottom: 2rem;
        box-shadow: 6px 4px 6px rgba(0, 0, 0, 0.1);
    }

    .upload-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .demo-section {
        padding: 2rem;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
        text-align: center;
    }

    .pipeline-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }

    /* Feature Cards */
    .feature-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
        height: 200px;
        width: 100%;
        overflow: hidden;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .feature-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        color: #2c3e50;
        text-align: center;
        width: 100%;
    }

    .feature-card p {
        margin: 0;
        font-size: 1rem;
        line-height: 1.2;
        color: #666;
        text-align: center;
        width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
    }

    /* Messages and Info Boxes */
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

    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .pipeline-step {
        color: #2c3e50;
    }

    .pipeline-step p {
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

def create_file_uploader():
    if 'file_name' not in st.session_state:
        st.session_state['file_name'] = None
    
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
                file_name = file_name.split('.')[0]
                st.markdown(f"""
                    <div class='success-message'>
                        ✅ File "{file_name}" uploaded successfully!
                    </div>
                """, unsafe_allow_html=True)
                st.session_state['file_name'] = file_name
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with col2:
        if uploaded_file is not None:
            with st.spinner("Processing image..."):
                st.image(uploaded_file, caption="Preview")
    
    return uploaded_file

def create_camera_input():
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
    
    return camera_input

def display_tips():
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
        
def info_box():
    st.markdown("""
        <div class='info-box'>
            Processing complete! Available options:
            - ▶️ Play the converted music
            - 💾 Download as MIDI file
            - 🎼 View the musical notation
        </div>
    """, unsafe_allow_html=True)
