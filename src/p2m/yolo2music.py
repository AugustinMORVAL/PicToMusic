from music21 import converter, braille, midi, instrument, tempo, environment, clef
from p2m.model import predict
from p2m.converter import XMLMEIConverter, CLEF_TO_TREBLE
import re
from io import BytesIO
import subprocess
import tempfile
import os

from PyQt5.QtCore import QLibraryInfo, QCoreApplication

# Si Qt n'a pas détecté automatiquement le bon chemin des plugins, on le configure manuellement
qt_plugins_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)

# Définir explicitement le chemin des plugins Qt avant d'importer PyQt5
if qt_plugins_path:
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugins_path
else:
    print("Erreur: impossible de détecter le chemin des plugins Qt.")

def inverse_transpose(clef: str, note_str: str) -> str:
    """
    Convert a note string from treble clef back to the original clef,
    using the inverse of the CLEF_TO_TREBLE mapping.
    
    Args:
        clef (str): The original clef label (e.g., "C3", "F3", etc.)
        note_str (str): The note string in treble clef to be converted.
        
    Returns:
        str: The note string converted back to the original clef.
    """
    mapping = CLEF_TO_TREBLE.get(clef)
    if mapping is None:
        # If there's no mapping for the given clef, return the original note string
        return note_str

    # Build the inverse mapping: treble note -> original clef note
    inverse_mapping = {treble: orig for orig, treble in mapping.items()}
    
    # Sort keys in descending order of length to match longer patterns first
    sorted_keys = sorted(inverse_mapping.keys(), key=len, reverse=True)
    pattern = '|'.join(re.escape(key) for key in sorted_keys)
    
    # Replace each occurrence of a treble note with its original clef note
    converted = re.sub(pattern, lambda m: inverse_mapping[m.group(0)], note_str)
    return converted

def abc_to_braille(abc_file):
    # Load ABC file
    abc_score = converter.parse(abc_file, format='abc')
    braille_rep = braille.translate.objectToBraille(abc_score)
    return braille_rep

def abc_to_midi_and_play(abc_file, output_file=None, play=False, instrument_class=None, tempo_bpm=None, musecore=False):
    # Load ABC file
    clef_abc = re.search(r'clef\s*=\s*(\S+)', abc_file, re.IGNORECASE).group(1)
    abc_score = converter.parse(abc_file, format='abc')
    
    part = abc_score.parts[0]
    desiredClef = clef.clefFromString(clef_abc)
    part.replace(part.getElementsByClass(clef.Clef)[0], desiredClef)

    if instrument_class:
        # Create a Stream directly and apply instrument to the elements in the stream
        abc_score.parts[0].insert(0, instrument_class())

    # Set the tempo if provided
    if tempo_bpm:
        tempo_marking = tempo.MetronomeMark(number=tempo_bpm)
        abc_score.insert(0, tempo_marking)

    if output_file:
        # Write the score to a MIDI file
        abc_score.write('midi', fp=output_file)
        print(f"MIDI file saved to {output_file}")
    
    if play:
        # Play the MIDI file
        player = midi.realtime.StreamPlayer(abc_score)
        player.play()

    if musecore:
        # Créer un fichier temporaire pour le fichier musescore
        with tempfile.NamedTemporaryFile(delete=False, suffix='.musicxml') as temp_file:
            # Exporter le score au format LilyPond dans ce fichier temporaire
            abc_score.write('musicxml', fp=temp_file.name)
            # os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/path/to/correct/qt/plugins/platforms"

            # Appeler LilyPond pour générer la sortie (généralement un PDF)
            subprocess.run(['/usr/bin/mscore3', temp_file.name])
    
def abc_to_image(abc_file, musescore_path='/usr/bin/mscore3'):
    # Parser le fichier ABC
    abc_score = converter.parse(abc_file, format='abc')

    # Show the musical score
    abc_score.show('lily.png')  # This uses MuseScore or LilyPond to render the score

def write_abc(notes: list, clef_abc: str = 'treble',  gamme: str = 'C', meter: str = '4/4'):
    """
    Writes a list of notes to an ABC file.
    :param notes: List of notes to write to the ABC file.
    """
    abc = 'X:1\n' + f'M:{meter}\n' + f'K:{gamme} clef={clef_abc}\n' + "L:1/16\n" +' '.join(notes)
    return abc

def yolo2abc(result):
    """
    Sorts YOLO predictions by x-coordinate and returns class names.

    :param result : YOLO prediction object
    :return: List of detected notes in sorted x order
    """
    gammes = {
        "0": "C", "1s": "G", "2s": "D", "3s": "A", "4s": "E", "5s": "B",
        "6s": "F#", "7s": "C#", "1f": "F", "2f": "Bb", "3f": "Eb",
        "4f": "Ab", "5f": "Db", "6f": "Gb", "7f": "Cb"
    }

    clef_abc_mapping = {
    "C1": "soprano",         # C clef on the 1st line
    "C2": "mezzosoprano",    # C clef on the 2nd line
    "C3": "alto",            # C clef on the 3rd line
    "C4": "tenor",           # C clef on the 4th line
    "G2": "treble",          # G clef on the 2nd line (standard treble)
    "F3": "baritone-f",      # F clef on the 3rd line
    "F4": "bass"             # F clef on the 4th line (standard bass)
    }

    # Convert tensors to lists
    cls_list = result.boxes.cls.tolist()
    data_list = result.boxes.data.tolist()
    class_names = result.names

    # Combine class IDs and bounding box info
    detections = list(zip(cls_list, data_list))

    # Sort detections by x-coordinate (first value in `data_list`)
    sorted_detections = sorted(detections, key=lambda x: x[1][0])

    # Get class names in sorted order
    sorted_notes = [class_names[int(d[0])] for d in sorted_detections]
    i=0
    if sorted_notes[0] in ['C1', 'C2', 'C3', 'C4', 'G2', 'F3', 'F4']:
        clef_abc=clef_abc_mapping[sorted_notes[0]]
        i+=1
    if sorted_notes[1] in gammes.values():
        gamme=sorted_notes[1]
        i+=1
    if bool(re.match(r'^\d+/\d+$', sorted_notes[2])):
        i+=1
        metrics=sorted_notes[2]

    if sorted_notes[0]!='G2':
        sorted_notes = [inverse_transpose(sorted_notes[0], note) for note in sorted_notes[i:]]
    else:
        sorted_notes = sorted_notes[i:]

    abc = write_abc(sorted_notes, clef_abc, gamme, metrics)

    return abc

if __name__ == "__main__":
    model_path = "models/chopinl.pt"
    image_path = "data/YOLO_small_1000/test/images/000105777-1_1_1.png"
    result = predict(model_path, image_path)[0]
    abc = yolo2abc(result)

    print(abc)


    # mei = XMLMEIConverter('resources/dataset/batch_1/match/labels/000100536-1_1_1.mei')
    # abc = mei.mei_to_abc()

    abc_to_midi_and_play(abc, play=True, instrument_class = instrument.AltoSaxophone, tempo_bpm=120, musecore=False)
    # abc_to_braille(abc)
    # abc_to_image(abc)
    # print(instrument.__dict__.keys()) 
