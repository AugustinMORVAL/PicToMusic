import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from sonatabene.converter.converter_yolo import yolo_to_abc, inverse_transpose
from sonatabene.converter.converter_abc import (
    abc_conversion, abc_to_midi, abc_to_braille, abc_to_musicxml,
    abc_to_pdf, abc_to_audio, abc_to_image, abc_to_musescore
)
from music21.stream import Stream
import os
import tempfile
from pathlib import Path
from music21.exceptions21 import Music21Exception
from midi2audio import FluidSynth

@pytest.fixture
def mock_yolo_result():
    """Fixture for creating a mock YOLO result"""
    mock_result = MagicMock()
    mock_result.boxes = MagicMock()
    mock_result.boxes.cls = np.array([0, 1, 2])  # Example class indices
    mock_result.boxes.data = np.array([
        [0, 0, 0, 0, 0, 0],  # x1, y1, x2, y2, conf, cls
        [1, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0]
    ])
    mock_result.names = {
        0: "G2",  # Treble clef
        1: "C",   # Key signature
        2: "4/4"  # Time signature
    }
    return mock_result

@pytest.fixture
def temp_abc_file():
    """Fixture for creating a temporary ABC file"""
    with tempfile.NamedTemporaryFile(suffix='.abc', delete=False) as temp_file:
        temp_file.write(b"""
X:1
T:Test
M:4/4
L:1/8
Q:1/4=120
K:C clef=treble
C D E F | G A B c |]
""")
        temp_file_path = temp_file.name
    
    yield temp_file_path
    
    # Cleanup
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)

@pytest.fixture
def temp_output_files():
    """Fixture for creating temporary output files"""
    files = {
        'midi': tempfile.NamedTemporaryFile(suffix='.mid', delete=False).name,
        'musicxml': tempfile.NamedTemporaryFile(suffix='.xml', delete=False).name,
        'pdf': tempfile.NamedTemporaryFile(suffix='.pdf', delete=False).name,
        'audio': tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    }
    
    yield files
    
    # Cleanup
    for file_path in files.values():
        if os.path.exists(file_path):
            os.unlink(file_path)

class TestConverterYolo:
    def test_yolo_to_abc_basic(self, mock_yolo_result):
        """Test basic YOLO to ABC conversion"""
        results = [mock_yolo_result]
        abc_output = yolo_to_abc(results)
        
        # Check if basic ABC structure is present
        assert "X:1" in abc_output
        assert "M:4/4" in abc_output
        assert "K:C" in abc_output
        assert "clef=treble" in abc_output

    @pytest.mark.parametrize("clef,note,expected", [
        ("C3", "C", "C"),      # Alto clef
        ("F4", "D", "D"),      # Bass clef 
        ("G2", "G", "G"),      # Treble clef
        ("C1", "A", "A"),      # Soprano clef
    ])
    def test_inverse_transpose(self, clef, note, expected):
        """Test note transposition from treble clef to other clefs"""
        result = inverse_transpose(clef, note)
        assert result == expected

class TestConverterABC:
    def test_abc_conversion(self, temp_abc_file):
        """Test basic ABC conversion"""
        with patch('sonatabene.converter.converter_abc.re.search') as mock_search:
            mock_search.return_value.group.return_value = "treble"
            score = abc_conversion(temp_abc_file)
            assert isinstance(score, Stream)
            assert len(score.parts) == 1

    def test_abc_to_midi(self, temp_abc_file, temp_output_files):
        """Test ABC to MIDI conversion"""
        with patch('sonatabene.converter.converter_abc.re.search') as mock_search:
            mock_search.return_value.group.return_value = "treble"
            with patch('sonatabene.converter.converter_abc.midi.realtime.StreamPlayer'):
                score = abc_to_midi(
                    temp_abc_file,
                    output_file=temp_output_files['midi'],
                    play=False
                )
                assert isinstance(score, Stream)
                assert os.path.exists(temp_output_files['midi'])

    def test_abc_to_braille(self, temp_abc_file):
        """Test ABC to Braille conversion"""
        with patch('sonatabene.converter.converter_abc.re.search') as mock_search:
            mock_search.return_value.group.return_value = "treble"
            braille_output = abc_to_braille(temp_abc_file)
            assert isinstance(braille_output, str)
            assert len(braille_output) > 0

    def test_abc_to_musicxml(self, temp_abc_file, temp_output_files):
        """Test ABC to MusicXML conversion"""
        with patch('sonatabene.converter.converter_abc.re.search') as mock_search:
            mock_search.return_value.group.return_value = "treble"
            abc_to_musicxml(
                temp_abc_file,
                temp_output_files['musicxml']
            )
            assert os.path.exists(temp_output_files['musicxml'])

    @pytest.mark.skipif(not os.path.exists('/usr/bin/lilypond'), reason="Lilypond not installed")
    def test_abc_to_pdf(self, temp_abc_file, temp_output_files):
        """Test ABC to PDF conversion"""
        with patch('sonatabene.converter.converter_abc.re.search') as mock_search:
            mock_search.return_value.group.return_value = "treble"
            abc_to_pdf(
                temp_abc_file,
                temp_output_files['pdf']
            )
            assert os.path.exists(temp_output_files['pdf'])

    def test_abc_to_audio(self, temp_abc_file, temp_output_files):
        """Test ABC to audio conversion"""
        with patch('sonatabene.converter.converter_abc.re.search') as mock_search:
            mock_search.return_value.group.return_value = "treble"
            with pytest.raises(Music21Exception) as exc_info:
                abc_to_audio(
                    temp_abc_file,
                    temp_output_files['audio'],
                    format='wav'
                )
            assert "cannot support output in this format yet: wav" in str(exc_info.value)

    def test_abc_to_image(self, temp_abc_file):
        """Test ABC to image conversion"""
        with patch('sonatabene.converter.converter_abc.re.search') as mock_search:
            mock_search.return_value.group.return_value = "treble"
            with patch('sonatabene.converter.converter_abc.music21.stream.Stream.show') as mock_show:
                abc_to_image(temp_abc_file)
                mock_show.assert_called_once()

    @pytest.mark.skipif(not os.path.exists('/usr/bin/mscore3'), reason="MuseScore not installed")
    def test_abc_to_musescore(self, temp_abc_file):
        """Test ABC to MuseScore conversion"""
        with patch('sonatabene.converter.converter_abc.re.search') as mock_search:
            mock_search.return_value.group.return_value = "treble"
            with patch('subprocess.run') as mock_run:
                abc_to_musescore(
                    temp_abc_file,
                    musescore_path='/usr/bin/mscore3'
                )
                mock_run.assert_called_once()

    @pytest.mark.parametrize("instrument,tempo", [
        ("piano", 120),
        ("violin", 100),
        ("flute", 150),
    ])
    def test_abc_conversion_with_different_instruments(self, temp_abc_file, instrument, tempo):
        """Test ABC conversion with different instruments and tempos"""
        with patch('sonatabene.converter.converter_abc.re.search') as mock_search:
            mock_search.return_value.group.return_value = "treble"
            score = abc_conversion(
                temp_abc_file,
                instrument=instrument,
                tempo_bpm=tempo
            )
            assert isinstance(score, Stream)
            assert len(score.parts) == 1 