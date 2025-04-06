import re
from typing import Dict, List, Optional

# Clef mapping for ABC notation
CLEF_ABC_MAPPING = {
    "C1": "soprano",         # C clef on the 1st line
    "C2": "mezzosoprano",    # C clef on the 2nd line
    "C3": "alto",            # C clef on the 3rd line
    "C4": "tenor",           # C clef on the 4th line
    "G2": "treble",          # G clef on the 2nd line (standard treble)
    "F3": "baritone-f",      # F clef on the 3rd line
    "F4": "bass"             # F clef on the 4th line (standard bass)
}

# Clef transposition mapping
CLEF_TO_TREBLE = {
    "C1": {"C": "G", "D": "A", "E": "B", "F": "C", "G": "D", "A": "E", "B": "F#"},
    "C2": {"C": "E", "D": "F#", "E": "G", "F": "A", "G": "B", "A": "C", "B": "D"},
    "C3": {"C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "A": "A", "B": "B"},
    "C4": {"C": "A", "D": "B", "E": "C", "F": "D", "G": "E", "A": "F#", "B": "G#"},
    "F3": {"C": "F", "D": "G", "E": "A", "F": "B", "G": "C", "A": "D", "B": "E"},
    "F4": {"C": "D", "D": "E", "E": "F#", "F": "G", "G": "A", "A": "B", "B": "C#"}
}

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
        return note_str

    # Build the inverse mapping: treble note -> original clef note
    inverse_mapping = {treble: orig for orig, treble in mapping.items()}
    
    # Sort keys in descending order of length to match longer patterns first
    sorted_keys = sorted(inverse_mapping.keys(), key=len, reverse=True)
    pattern = '|'.join(re.escape(key) for key in sorted_keys)
    
    # Replace each occurrence of a treble note with its original clef note
    converted = re.sub(pattern, lambda m: inverse_mapping[m.group(0)], note_str)
    return converted

def yolo_to_abc(results):
    """
    Converts multiple YOLO predictions into ABC notation format.

    :param results: List of YOLO prediction objects, one for each staff line
    :return: ABC notation string
    """
    gammes = {
        "0": "C", "1s": "G", "2s": "D", "3s": "A", "4s": "E", "5s": "B",
        "6s": "F#", "7s": "C#", "1f": "F", "2f": "Bb", "3f": "Eb",
        "4f": "Ab", "5f": "Db", "6f": "Gb", "7f": "Cb"
    }

    if not results:
        return "No notes detected in the image."

    # Initialize ABC header
    abc_content = [
        "X:1",          # Reference number
        "T:",           # Title (blank for now)
        "M:4/4",        # Default time signature
        "L:1/8",        # Default note length
        "Q:1/4=120",    # Default tempo
        "K:C",          # Default key signature
    ]

    for i, result in enumerate(results):
        cls_list = result.boxes.cls.tolist()
        data_list = result.boxes.data.tolist()
        class_names = result.names

        if not cls_list or not data_list:
            continue

        detections = list(zip(cls_list, data_list))
        sorted_detections = sorted(detections, key=lambda x: x[1][0])
        sorted_notes = [class_names[int(d[0])] for d in sorted_detections]

        if i == 0:
            # Handle clef
            if len(sorted_notes) > 0 and sorted_notes[0] in CLEF_ABC_MAPPING:
                clef = sorted_notes[0]
                abc_content.append(f"%%clef {CLEF_ABC_MAPPING[clef]}")

            # Handle key signature
            if len(sorted_notes) > 1 and sorted_notes[1] in gammes.values():
                key = sorted_notes[1]
                abc_content[5] = f"K:{key}"

            # Handle time signature
            if len(sorted_notes) > 2 and bool(re.match(r'^\d+/\d+$', sorted_notes[2])):
                time_sig = sorted_notes[2]
                abc_content[2] = f"M:{time_sig}"

        # Process notes for this line
        notes = sorted_notes[3:] if len(sorted_notes) > 3 else []
        if notes:
            measure = []
            current_measure = []
            for note in notes:
                current_measure.append(note)
                if len(current_measure) >= 8:  # 8 eighth notes = 4 beats
                    measure.append(' '.join(current_measure))
                    current_measure = []
            
            if current_measure:
                measure.append(' '.join(current_measure))
            
            if i == 0 and sorted_notes[0] in CLEF_TO_TREBLE:
                measure = [inverse_transpose(sorted_notes[0], m) for m in measure]
            
            abc_content.extend(measure)

    abc_content.append('|]')

    return '\n'.join(abc_content)