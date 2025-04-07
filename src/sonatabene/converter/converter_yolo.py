import re
from typing import Dict, List, Optional
from sonatabene.converter.mapping import CLEF_TO_TREBLE, CLEF_ABC_MAPPING, GAMMES

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

def group_and_sort_detections(
    detections,
    y_tolerance: Optional[float] = None,
    tolerance_factor: float = 0.2
):
    """
    Regroupe les détections par ligne (en fonction de y_center), puis trie chaque ligne par x_center.
    Le y_tolerance est automatiquement estimé si non fourni.
    
    Args:
        detections: Liste de tuples (label, (x_center, y_center, width, height))
        y_tolerance: Tolérance verticale pour considérer deux boîtes sur la même ligne
        tolerance_factor: Multiplicateur de la hauteur moyenne pour estimer y_tolerance
        
    Returns:
        Liste triée des détections, ligne par ligne.
    """

    if not detections:
        return []

    if y_tolerance is None:
        avg_height = sum(d[1][3] for d in detections) / len(detections)
        y_tolerance = avg_height * tolerance_factor

    detections = sorted(detections, key=lambda d: d[1][1])

    lines = []
    current_line = []

    for det in detections:
        _, (_, y_center, _, _, _, _) = det

        if not current_line:
            current_line.append(det)
        else:
            _, (_, ref_y, _, _, _, _) = current_line[0]
            if abs(y_center - ref_y) < y_tolerance:
                current_line.append(det)
            else:
                current_line = sorted(current_line, key=lambda d: d[1][0])
                lines.append(current_line)
                current_line = [det]

    if current_line:
        current_line = sorted(current_line, key=lambda d: d[1][0])
        lines.append(current_line)

    return [d for line in lines for d in line]

def yolo_to_abc(results):
    """
    Converts multiple YOLO predictions into ABC notation format.

    :param results: List of YOLO prediction objects, one for each staff line
    :return: ABC notation string
    """

    if not results:
        return "No notes detected in the image."

    # Initialize ABC header
    abc_content = [
        "X:1",          # Reference number
        "T:",           # Title (blank for now)
        "M:4/4",        # Default time signature
        "L:1/16",        # Default note length
        "Q:1/4=120",    # Default tempo
        "K:C clef=G2",          # Default key signature
    ]

    for i, result in enumerate(results):
        cls_list = result.boxes.cls.tolist()
        data_list = result.boxes.data.tolist()
        class_names = result.names

        if not cls_list or not data_list:
            continue

        detections = list(zip(cls_list, data_list))
        sorted_detections = group_and_sort_detections(detections)
        sorted_notes = [class_names[int(d[0])] for d in sorted_detections]

        if i == 0:
            # Handle key signature
            if len(sorted_notes) > 1 and sorted_notes[1] in GAMMES.values():
                key = sorted_notes[1]
                abc_content[5] = f"K:{key} clef={CLEF_ABC_MAPPING.get(sorted_notes[0], 'treble')}"

            # Handle time signature
            if len(sorted_notes) > 2 and bool(re.match(r'^\d+/\d+$', sorted_notes[2])):
                time_sig = sorted_notes[2]
                abc_content[2] = f"M:{time_sig}"

        # Process notes for this line
        notes = [n for n in sorted_notes if n not in GAMMES.values() and n not in CLEF_ABC_MAPPING and not bool(re.match(r'^\d+/\d+$', n))] if len(sorted_notes) > 3 else []
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