# 🎼 SonataBene: Music Score to Audio

SonataBene is an advanced computer vision application that transforms sheet music into playable MIDI files. Using a combination of traditional image processing techniques and deep learning models, it accurately detects and interprets musical notation from both digital images and camera captures.

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

1. **Image Processing Pipeline** (`src/sonatabene/parser.py`)
   - Traditional CV techniques for staff line detection and note segmentation
   - Mathematical morphology for staff line detection
   - Contour analysis for note component detection
   - Geometric feature-based symbol recognition

2. **Deep Learning Models** (`models/`)
   - YOLOv11 for musical element detection
   - Fine-tuned model for note recognition
   - Ensemble approach combining traditional and deep learning methods

3. **Music Generation**
   - ABC notation conversion
   - MIDI file generation with accurate timing and pitch
   - Support for various instruments and styles

### Training Pipeline

1. **Initial Dataset (250 Staffs)**
   - Traditional algorithmic model for staff line detection and note segmentation
   - Generated annotations for first YOLOv11 model training
   - Focus on general musical element detection

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
│   └── sonatabene/
│       ├── parser.py      # Core image processing pipeline
│       ├── model.py       # Deep learning model definitions
│       ├── labelizer.py   # Data labeling utilities
│       ├── cli.py         # Command line interface
│       ├── utils.py       # Utility functions
│       ├── scoretyping.py # Score type definitions
│       └── converter/     # Music format conversion utilities
├── UI/
│   ├── statics.py        # UI static elements
│   └── pparser_app_logic.py # Application logic
├── models/               # Trained model weights
├── configs/             # Configuration files
├── data/                # Training and test data
├── tests/              # Test suite
├── notebooks/          # Development notebooks
├── documentation/      # Project documentation
├── app.py             # Main application entry point
└── requirements.txt    # Python dependencies
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

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install external dependencies:

### macOS
```bash
# Install Lilypond (for PDF generation)
brew install lilypond

# Install MuseScore (for MuseScore format conversion)
brew install musescore
```

### Windows
- Download and install [Lilypond](https://lilypond.org/download.html)
- Download and install [MuseScore](https://musescore.org/en/download)

### Linux
```bash
# Ubuntu/Debian
sudo apt install lilypond musescore

# Fedora
sudo dnf install lilypond musescore
```

## 💻 Usage

### Web Interface
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

### Command Line Interface

The CLI provides a simple interface to convert sheet music to various audio formats. Here's how to use it:

#### Basic Usage

```bash
# Convert sheet music to MIDI
snb music play --image-path sheet.jpg --output-format midi --output-file output.mid

# Convert sheet music to MusicXML
snb music play --image-path sheet.jpg --output-format musicxml --output-file output.musicxml

# Convert sheet music to Audio (WAV/MP3)
snb music play --image-path sheet.jpg --output-format wav --output-file output.wav
```

#### Advanced Usage

```bash
# Train a custom model
snb model train \
    --data-path data/dataset.yaml \
    --model-path models/pretrained.pt \
    --config-path configs/training_config.yaml

# Predict notes from an image
snb model predict \
    --image-path sheet.jpg \
    --model-path models/chopin.pt \
    --config-path configs/predict_config.yaml

# Customize the output with advanced options
snb music play \
    --image-path sheet.jpg \
    --instrument Violin \
    --tempo 100 \
    --dynamics '{"p": 40, "f": 100}' \
    --articulation '{"staccato": 0.5}' \
    --output-format midi \
    --output-file output.mid
```

#### Command Options

| Command | Option | Description | Default |
|---------|--------|-------------|---------|
| **model train** | `--data-path` | Path to dataset configuration file | (required) |
| | `--model-path` | Path to initial model weights | 'yolo11n.pt' |
| | `--config-path` | Path to training configuration | 'configs/training_config.yaml' |
| **model predict** | `--image-path` | Path to sheet music image | (required) |
| | `--model-path` | Path to trained model | 'models/chopin.pt' |
| | `--config-path` | Path to prediction configuration | 'configs/predict_config.yaml' |
| **music play** | `--image-path` | Path to sheet music image | (required) |
| | `--model-path` | Path to trained model | 'models/chopin.pt' |
| | `--instrument` | Output instrument | 'Piano' |
| | `--tempo` | Tempo in BPM | 120 |
| | `--dynamics` | Dynamic markings (JSON) | '{"p": 40, "f": 100}' |
| | `--articulation` | Articulation settings (JSON) | '{"staccato": 0.5}' |
| | `--output-format` | Output format (midi/musicxml/pdf/wav/mp3) | 'midi' |
| | `--output-file` | Path to save output | (required) |

## 📚 Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [YOLOv11 Documentation](https://github.com/ultralytics/yolov11)
- [Music21 Documentation](http://web.mit.edu/music21/doc/)
- [ABC Notation Guide](http://abcnotation.com/wiki/abc:standard)
- [MIDI File Format Specification](https://www.midi.org/specifications)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV team for the computer vision library
- Ultralytics for the YOLOv11 implementation
- Music21 team for music processing tools
- All contributors and users of the project
