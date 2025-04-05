from music21 import converter, braille, midi, instrument, tempo
from music21.instrument import Piano, Violin, Viola, Violoncello, Contrabass, Guitar, Harp, \
    PanFlute, Flute, Piccolo, Clarinet, Oboe, Bassoon, Saxophone, \
    Trumpet, Trombone, Horn, Tuba, \
    Timpani, Percussion, \
    Choir, Organ, Harpsichord, Celesta, Glockenspiel, Xylophone, Marimba, Vibraphone
import music21.stream
import music21.instrument as m21_instrument
from p2m.model import predict
import p2m.converter.converter_yolo as converter_yolo
from typing import Union, Dict, List, Optional
from io import BytesIO
from pathlib import Path
import loguru

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

def abc_to_braille(abc_file, instrument: Optional[Union[str, type]] = Piano,
                  tempo_bpm: Optional[int] = 120, dynamics: Optional[Dict[str, int]] = None,
                  articulation: Optional[Dict[str, float]] = None):
    """Convert ABC notation to Braille."""
    abc_score = abc_conversion(abc_file, instrument, tempo_bpm, dynamics, articulation)
    return braille.translate.objectToBraille(abc_score)

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
if __name__ == "__main__":
    import cv2
    from p2m.parser import PParser

    image_path = "resources/samples/mary.png"
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
    abc_to_midi(abc, instrument_class = instrument.PanFlute, play=True)
    # abc_to_braille(abc)
    # abc_to_image(abc)
    # print(instrument.__dict__.keys()) 
