from music21 import converter, braille, midi, instrument, tempo, clef
from music21.instrument import Piano, Violin, Viola, Violoncello, Contrabass, Guitar, Harp, \
    PanFlute, Flute, Piccolo, Clarinet, Oboe, Bassoon, Saxophone, \
    Trumpet, Trombone, Horn, Tuba, \
    Timpani, Percussion, \
    Choir, Organ, Harpsichord, Celesta, Glockenspiel, Xylophone, Marimba, Vibraphone
import music21.stream
from p2m.model import predict
import p2m.converter.converter_yolo as converter_yolo
from typing import Union, Dict, Optional
from io import BytesIO
from pathlib import Path
import loguru
import tempfile
import subprocess
import os
import re

INSTRUMENT_MAP = {
    # Piano family
    'piano': Piano,
    'keyboard': Piano,
    
    # Strings
    'violin': Violin,
    'viola': Viola,
    'cello': Violoncello,
    'doublebass': Contrabass,
    'guitar': Guitar,
    'harp': Harp,
    
    # Woodwinds
    'flute': Flute,
    'piccolo': Piccolo,
    'clarinet': Clarinet,
    'oboe': Oboe,
    'bassoon': Bassoon,
    'saxophone': Saxophone,
    'panflute': PanFlute,
    
    # Brass
    'trumpet': Trumpet,
    'trombone': Trombone,
    'horn': Horn,
    'tuba': Tuba,
    
    # Percussion
    'timpani': Timpani,
    'drums': Percussion,
    'percussion': Percussion,
    
    # Other
    'voice': Choir,
    'choir': Choir,
    'organ': Organ,
    'harpsichord': Harpsichord,
    'celesta': Celesta,
    'glockenspiel': Glockenspiel,
    'xylophone': Xylophone,
    'marimba': Marimba,
    'vibraphone': Vibraphone
}

class ConverterError(Exception):
    """Base exception for converter-related errors."""
    pass


def abc_conversion(abc_file: Union[str, Path], 
                    instrument: Optional[Union[str, type]] = Piano,
                    tempo_bpm: Optional[int] = 120,
                    dynamics: Optional[Dict[str, int]] = None,
                    articulation: Optional[Dict[str, float]] = None) -> music21.stream.Stream:
    """
    Core function for ABC conversion with common parameters.
    
    Args:
        abc_file: ABC notation string or file path
        output_file: Path to save MIDI file (optional)
        play: Whether to play the MIDI output
        instrument: Single instrument class (string name or type). Available instruments:
            - Piano family: piano, keyboard
            - Strings: violin, viola, cello, doublebass, guitar, harp
            - Woodwinds: flute, piccolo, clarinet, oboe, bassoon, saxophone, panflute
            - Brass: trumpet, trombone, horn, tuba
            - Percussion: timpani, drums, percussion
            - Other: voice, choir, organ, harpsichord, celesta, glockenspiel, xylophone, marimba, vibraphone
        tempo_bpm: Tempo in beats per minute
        dynamics: Dictionary of dynamic markings
        articulation: Dictionary of articulation settings
        
    Returns:
        abc_score: music21 object
    """
    try:
        abc_score = converter.parse(abc_file, format='abc')

        clef_abc = re.search(r'clef\s*=\s*(\S+)', abc_file, re.IGNORECASE).group(1)
        
        part = abc_score.parts[0]
        desiredClef = clef.clefFromString(clef_abc)
        part.replace(part.getElementsByClass(clef.Clef)[0], desiredClef)

        if isinstance(instrument, str):
            instrument = instrument.lower()
            if instrument in INSTRUMENT_MAP:
                instrument_class = INSTRUMENT_MAP[instrument]
            else:
                loguru.logger.warning(f"Instrument '{instrument}' not found in mapping, defaulting to Piano")
                instrument_class = Piano
        elif isinstance(instrument, type):
            instrument_class = instrument
        else:
            instrument_class = Piano

        abc_score.parts[0].insert(0, instrument_class())

        if dynamics:
            apply_dynamics(abc_score, dynamics)

        if articulation:
            apply_articulation(abc_score, articulation)

        if tempo_bpm:
            if not isinstance(tempo_bpm, int) or tempo_bpm <= 0:
                raise ValueError("Tempo must be a positive integer")
            tempo_marking = tempo.MetronomeMark(number=tempo_bpm)
            abc_score.insert(0, tempo_marking)

        return abc_score

    except Exception as e:
        raise ConverterError(f"Error in core conversion: {str(e)}")

def abc_to_midi(abc_file: Union[str, Path], output_file: Optional[Union[str, Path, BytesIO]] = None, 
           play: bool = False, instrument: Optional[Union[str, type]] = Piano, 
           tempo_bpm: Optional[int] = 120, dynamics: Optional[Dict[str, int]] = None, 
           articulation: Optional[Dict[str, float]] = None):
    """
    Convert ABC notation to MIDI with enhanced features and validation.
    
    Args:
        abc_file: ABC notation string or file path
        output_file: Path to save MIDI file (optional) or BytesIO buffer
        play: Whether to play the MIDI output
        instrument: Single instrument class (string name or type)
        tempo_bpm: Tempo in beats per minute
        dynamics: Dictionary of dynamic markings
        articulation: Dictionary of articulation settings
        
    Returns:
        abc_score: abc_score object
    """
    try:
        abc_score = abc_conversion(abc_file, instrument, tempo_bpm, dynamics, articulation)

        if output_file:
            if isinstance(output_file, BytesIO):
                mf = midi.translate.streamToMidiFile(abc_score)
                midi_data = mf.writestr()
                output_file.write(midi_data)
            else:
                abc_score.write('midi', fp=output_file)
                print(f"MIDI file saved to {output_file}")
        
        if play:
            player = midi.realtime.StreamPlayer(abc_score)
            loguru.logger.info("Playing...")
            player.play()
            
        return abc_score

    except Exception as e:
        raise ConverterError(f"Error converting to MIDI: {str(e)}")

def abc_to_braille(abc_file: Union[str, Path], 
                  instrument: Optional[Union[str, type]] = Piano,
                  tempo_bpm: Optional[int] = 120,
                  output_file: Optional[Union[str, Path]] = None) -> str:
    """
    Convert ABC notation to Braille music notation.
    
    Args:
        abc_file: ABC notation string or file path
        instrument: Instrument to use
        tempo_bpm: Tempo in beats per minute
        output_file: Optional path to save the Braille output
        
    Returns:
        str: Braille music notation
    """
    try:
        abc_score = abc_conversion(abc_file, instrument, tempo_bpm)
        braille_rep = braille.translate.objectToBraille(abc_score)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(braille_rep)
            print(f"Braille notation saved to {output_file}")
            
        return braille_rep
        
    except Exception as e:
        raise ConverterError(f"Error converting to Braille: {str(e)}")

def abc_to_musicxml(abc_file, output_file, instrument: Optional[Union[str, type]] = Piano,
                   tempo_bpm: Optional[int] = 120, dynamics: Optional[Dict[str, int]] = None,
                   articulation: Optional[Dict[str, float]] = None):
    """Convert ABC notation to MusicXML format."""
    abc_score = abc_conversion(abc_file, instrument, tempo_bpm, dynamics, articulation)
    abc_score.write('musicxml', fp=output_file)
    print(f"MusicXML file saved to {output_file}")

def abc_to_pdf(abc_file, output_file, instrument: Optional[Union[str, type]] = Piano,
              tempo_bpm: Optional[int] = 120, dynamics: Optional[Dict[str, int]] = None,
              articulation: Optional[Dict[str, float]] = None):
    """Convert ABC notation to PDF score."""
    abc_score = abc_conversion(abc_file, instrument, tempo_bpm, dynamics, articulation)
    abc_score.write('lily.pdf', fp=output_file)
    print(f"PDF score saved to {output_file}")

def abc_to_audio(abc_file, output_file, format='wav', instrument: Optional[Union[str, type]] = Piano,
                tempo_bpm: Optional[int] = 120, dynamics: Optional[Dict[str, int]] = None,
                articulation: Optional[Dict[str, float]] = None):
    """Convert ABC notation to audio file."""
    abc_score = abc_conversion(abc_file, instrument, tempo_bpm, dynamics, articulation)
    abc_score.write(format, fp=output_file)
    print(f"Audio file saved to {output_file}")

def abc_to_image(abc_file, instrument: Optional[Union[str, type]] = Piano,
                tempo_bpm: Optional[int] = 120, dynamics: Optional[Dict[str, int]] = None,
                articulation: Optional[Dict[str, float]] = None):
    """Convert ABC notation to image."""
    abc_score = abc_conversion(abc_file, instrument, tempo_bpm, dynamics, articulation)
    abc_score.show('musicxml.png')


def apply_dynamics(score, dynamics):
    """Apply dynamic markings to the score."""
    default_dynamics = {
        'ppp': 20, 'pp': 30, 'p': 40, 'mp': 50,
        'mf': 60, 'f': 70, 'ff': 80, 'fff': 90
    }
    dynamics = {**default_dynamics, **dynamics}
    
    for note in score.recurse().notes:
        if hasattr(note, 'expressions'):
            for exp in note.expressions:
                if exp.name in dynamics:
                    note.volume.velocity = dynamics[exp.name]

def apply_articulation(score, articulation):
    """Apply articulation markings to the score."""
    default_articulation = {
        'staccato': 0.5,  # 50% of note duration
        'tenuto': 1.0,    # Full duration
        'accent': 1.2     # 120% of note duration
    }
    articulation = {**default_articulation, **articulation}
    
    for note in score.recurse().notes:
        if hasattr(note, 'articulations'):
            for art in note.articulations:
                if art.name in articulation:
                    note.duration.quarterLength *= articulation[art.name]

def abc_to_musescore(abc_file: Union[str, Path], output_file: Optional[Union[str, Path]] = None,
                    instrument: Optional[Union[str, type]] = Piano,
                    tempo_bpm: Optional[int] = 120,
                    musescore_path: str = r'/mnt/c/Program Files/MuseScore 4/bin/MuseScore4.exe',
                    open: bool = False) -> None:
    """
    Convert ABC notation to MuseScore format and optionally display it.
    
    Args:
        abc_file: ABC notation string or file path
        output_file: Path to save the MuseScore file (optional)
        instrument: Instrument to use
        tempo_bpm: Tempo in beats per minute
        musescore_path: Path to MuseScore executable
        open: Whether to open the file in MuseScore directly
    """
    try:
        # ABC conversion (replace this with your actual ABC conversion logic)
        abc_score = abc_conversion(abc_file, instrument, tempo_bpm)
        
        if output_file or open:
            # Create a temporary MusicXML file
            temp_xml = tempfile.NamedTemporaryFile(delete=False, suffix='.musicxml')
            abc_score.write('musicxml', fp=temp_xml.name)
            
            if open:
                # Create a temporary MSCZ file
                temp_mscz = tempfile.NamedTemporaryFile(delete=False, suffix='.mscz')
                subprocess.run([musescore_path, temp_xml.name, '-o', temp_mscz.name])
                subprocess.run([musescore_path, temp_mscz.name])  # Open MuseScore with the file
                os.unlink(temp_xml.name)  # Clean up temporary files
                os.unlink(temp_mscz.name)
            else:
                # Save the file to the provided output path
                subprocess.run([musescore_path, temp_xml.name, '-o', str(output_file)])
                print(f"MuseScore file saved to {output_file}")
                os.unlink(temp_xml.name)  # Clean up temporary file
        else:
            abc_score.show('musicxml.png')
            
    except Exception as e:
        raise Exception(f"Error converting to MuseScore: {str(e)}")

if __name__ == "__main__":
    import cv2
    from p2m.parser import PParser

    image_path = "resources/samples/mary.jpg"
    loguru.logger.info("Predicting...")
    parser = PParser()
    parser.load_image(image_path)
    stafflines = parser.find_staff_lines(min_contour_area=10000)
    staffs = [cv2.cvtColor(staffline.image, cv2.COLOR_RGB2BGR) for staffline in stafflines]
    results = list(predict(staffs[0], model_path="models/chopin.pt"))
    loguru.logger.info("Converting to ABC...")
    abc = converter_yolo.yolo_to_abc(results)
    loguru.logger.info(f"ABC converted : \n{abc}")

    loguru.logger.info("Converting to MIDI...")
    abc_to_midi(abc, instrument = PanFlute, play=True)
    # abc_to_musescore(abc, instrument = instrument.PanFlute)
    # abc_to_braille(abc)
    # abc_to_image(abc)
    # print(instrument.__dict__.keys()) 
