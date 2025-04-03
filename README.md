# 🎼 PicToMusic: Sheet Music to MIDI Converter

PicToMusic is an advanced computer vision application that transforms sheet music into playable MIDI files. Using a combination of traditional image processing techniques and deep learning models, it accurately detects and interprets musical notation from both digital images and camera captures.

## 🎯 Project Overview

The project is built with a modular architecture that combines multiple approaches for robust musical notation recognition:

1. **Traditional Image Processing Pipeline**
   - Initial preprocessing and enhancement of sheet music images
   - Staff line detection using mathematical morphology
   - Note component detection through contour analysis
   - Basic musical symbol recognition using geometric features

2. **Deep Learning Integration**
   - YOLOv11 model fine-tuned using preprocessed data from traditional pipeline
   - Specialized note recognition model for accurate pitch and duration classification
   - Ensemble approach combining traditional and deep learning methods

3. **Music Generation System**
   - Conversion to ABC notation format
   - MIDI file generation with accurate timing and pitch
   - Support for various musical instruments and styles

## 🔬 Technical Implementation

### Architecture Overview

```
┌─────────────────┐
│   Sheet Music   │
│     Image       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌───────────────────┐
│  Preprocessing  │      │   Image Quality   │
│  • Grayscale    ├─────►│   Enhancement     │
│  • Thresholding │      │   • Noise removal │
└────────┬────────┘      │   • Contrast adj. │
         │               └─────────┬─────────┘
         ▼                         │
    ┌────┴─────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│        Parallel Processing           │
│                                      │
│    ┌───────────┐      ┌──────────┐   │
│    │Staff Line │      │  Note    │   │
│    │Detection  │      │Detection │   │
│    └─────┬─────┘      └────┬─────┘   │
│          │                 │         │
│          ▼                 ▼         │
│    ┌──────────┐      ┌────────────┐  │
│    │Staff Line│      │  YOLOv11   │  │
│    │Detection │      │  Element   │  │
│    └────┬─────┘      │ Detection  │  │
│         │            └─────┬──────┘  │
│         │                  │         │
│         │                  ▼         │
│         │            ┌────────────┐  │
│         │            │  YOLOv11   │  │
│         │            │  Note      │  │
│         │            │ Recognition│  │
│         │            └─────┬──────┘  │
└─────────┼──────────────────┼─────────┘
          │                  │ 
          └───────┐     ┌────┘
                  │     │
                  ▼     ▼
         ┌─────────────────────┐
         │    Result Fusion    │
         │  • Confidence Score │
         │  • Ensemble Method  │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   ABC Notation      │
         │   Generation        │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   MIDI Output       │
         │   Generation        │
         └─────────────────────┘
```

### Core Components

#### 1. Image Processing Pipeline (`src/p2m/parser.py`)
- Traditional computer vision techniques for initial processing
- Staff line detection using mathematical morphology
- Note component detection through contour analysis
- Basic musical symbol recognition

#### 2. Deep Learning Models (`models/`)
- YOLOv11 model for musical element detection
- Fine-tuned note recognition model
- Ensemble method combining multiple models

#### 3. Music Generation
- ABC notation conversion
- MIDI file generation
- Timing and pitch accuracy optimization

### Construction Process

#### 1. First Database Construction (250 Staffs)
- Initial dataset of 250 musical staffs collected
- Traditional algorithmic model used for:
  - Staff line detection
  - Note segmentation
  - Basic musical symbol recognition
- Generated annotations used to train first YOLOv11 model
- This model learns to detect general musical elements

#### 2. Second Database Construction
- First YOLOv11 model used to process new sheet music
- Generated bounding boxes and segmentations
- Manual verification and correction of detections
- Creation of cropped note images with accurate labels
- Database used to fine-tune second YOLOv11 model
- This model specializes in precise note recognition

#### Training Pipeline
```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│                           Training Process                        │
│                                                                   │ 
│  Detection Model Training          Classification Model Training  │
│                                                                   │
│  ┌─────────────────┐                                              │
│  │  Algorithmic    │                                              │
│  │  Model          │                                              │
│  │  • Staff Lines  │                                              │
│  │  • Segmentation │                                              │
│  └────────┬────────┘                                              │
│           │                                                       │
│           ▼                        ┌──────────────────┐           │
│  ┌─────────────────┐               │  MEI Extraction  │           │
│  │  Detection      │               │  • Note Labels   │           │
│  │  YOLOv11        │               │  • Pitch Info    │           │
│  │  Training       │               │  • Duration      │           │
│  │  • 250 Staffs   │               └────────┬─────────┘           │
│  │  • Element Det. │                        │                     │
│  └────────┬────────┘                        │                     │
│           │                                 │                     │      
│           └────────────────┐    ┌───────────┘                     │
│                            ▼    ▼                                 │
│                     ┌─────────────────┐                           │
│                     │  BBox Gen. &    │                           │
│                     │  Verification   │                           │
│                     │  • Manual Check │                           │
│                     │  • Corrections  │                           │
│                     └────────┬────────┘                           │
│                              │                                    │
│                              ▼                                    │
│                     ┌─────────────────┐                           │
│                     │ Classification  │                           │
│                     │    YOLOv11      │                           │
│                     │  Training       │                           │
│                     │  • 30k Staffs   │                           │
│                     │  • Note Rec.    │                           │
│                     └─────────────────┘                           │
└───────────────────────────────────────────────────────────────────┘
```

## 🛠️ Project Structure

```
PicToMusic/
├── src/
│   └── p2m/
│       ├── parser.py      # Image processing pipeline
│       ├── model.py       # Deep learning model definitions
│       ├── mei2abc.py     # Music format converter
│       └── utils.py       # Utility functions
├── models/
│   ├── yparser.pt        # YOLOv11 Detection model weights
│   └── note_recognition/ # Note recognition model
├── UI/                   # Web interface
├── tests/               # Test suite
└── notebooks/           # Development notebooks
```

## 🔧 Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/AugustinMORVAL/PicToMusic
cd PicToMusic
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Start the web interface:
```bash
streamlit run app.py
```

2. Upload or capture sheet music:
   - Support for PNG, JPG, JPEG formats
   - Real-time camera capture available
   - Automatic image enhancement

3. Process the image:
   - The system will automatically:
     - Preprocess the image
     - Detect staff lines and notes
     - Apply deep learning models
     - Generate MIDI output

4. Download the generated MIDI file

## 🔍 Technical Details

### Image Processing Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| Image Resolution | Max dimension | 1200px | 800-2000px |
| Staff Line Dilation | Line detection sensitivity | 3 | 1-10 |
| Note Dilation | Note detection sensitivity | 2 | 1-10 |
| Min Staff Area | Minimum staff line size | 10000 | 1000-20000 |
| Min Note Area | Minimum note size | 50 | 10-1000 |
| Overlap Threshold | Component grouping threshold | 0.5 | 0.1-0.9 |

### Deep Learning Models

1. **YOLOv11 Musical Element Detection**
   - Input: Preprocessed image
   - Output: Bounding boxes for musical elements
   - Classes: Notes, Clefs, Time Signatures, etc.

2. **Note Recognition Model**
   - Input: Cropped note images
   - Output: Note type, pitch, and duration
   - Architecture: Custom CNN with attention mechanism

## 📚 Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [YOLOv11 Documentation](https://github.com/ultralytics/yolov11)
- [Music21 Documentation](http://web.mit.edu/music21/doc/)
- [ABC Notation Guide](http://abcnotation.com/wiki/abc:standard)
- [MIDI File Format Specification](https://www.midi.org/specifications)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
