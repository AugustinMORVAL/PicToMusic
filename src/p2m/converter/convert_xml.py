import re
from zipfile import ZipFile
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from lxml import etree as ET
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from abc import ABC
import os 
import time
import pandas as pd
from tqdm import tqdm

# Clef transposition mappings
CLEF_TO_TREBLE = {
    "C1": {  # C on the 1st line
        'a,': 'c',  'b,': 'd',  'c': 'e',  'd': 'f',  'e': 'g',  
        'f': 'a',  'g': 'b',  'a': "c'",  'b': "d'",
    },
    "C2": {  # C on the 2nd line
        'f,': 'c',  'g,': 'd',  'a,': 'e',  'b,': 'f',  'c': 'g',  
        'd': 'a',  'e': 'b',  'f': "c'",  'g': "d'",  'a': "e'",  'b': "f'",
    },
    "C3": {  # C on the 3rd line
        'd,': 'c',  'e,': 'd',  'f,': 'e',  'g,': 'f',  'a,': 'g',  
        'b,': 'a',  'c': 'b',  'd': "c'",  'e': "d'",  'f': "e'",  'g': "f'",  'a': "g'",  'b': "a'",
    },
    "C4": {  # C on the 4th line
        'b,,': 'c',  'c,': 'd',  'd,': 'e',  'e,': 'f',  'f,': 'g',  
        'g,': 'a',  'a,': 'b',  'b,': "c'",  'c': "d'",  'd': "e'",  'e': "f'",  'f': "g'",  'g': "a'",  'a': "b'",  'b': "c''",
    },
    "F4": {  # F on the 4th line
        'e,,': 'c',  'f,,': 'd',  'g,,': 'e',  'a,,': 'f',  'b,,': 'g',  'c,': 'a',  
        'd,': 'b', 'e,': "c'",  'f,': "d'",  'g,': "e'",  'a,': "f'",  'b,': "g'",  'c': "a'",  'd': "b'",  'e': "c''",  'f': "d''",  'g': "e''",  'a': "f''",  'b': "g''"
    },
    "F3": {  # F on the 3rd line
        'e,,':'a,',  'f,,': 'b,',  'g,,': 'c',  'a,,': 'd',  'b,,': 'e',  'c,': 'f',  
        'd,': 'g', 'e,': 'a',  'f,': 'b',  'g,': "c'",  'a,': "d'",  'b,': "e'",  'c': "f'",  'd': "g'",  'e': "a'",  'f': "b'",  'g': "c''",  'a': "d''",  'b': "e''"
    },
}


@dataclass
class ScoreDefinition:
    """Data class to hold score definition information."""
    key: str = ""
    meter_count: int = 4
    meter_unit: int = 4
    clef: str = "G2"


class BaseMEIConverter(ABC):
    """
    Base class for MEI to ABC converters with shared functionality.
    """
    
    # Static mappings shared across all converter implementations
    OCTAVES: Dict[int, str] = {0: ",,,,,", 1: ",,,,", 2: ",,,", 3: ",,", 4: ",", 5: "", 6: "'", 7: "''"}
    DURATION_MAPPING: Dict[str, str] = {
        '128': "///", '64': "//", '32': "/", '16': "1", '8': "2", '4': "4", 
        '2': "8", '1': "16", 'breve': "32", 'long': "64"
    }
    ACCID_MAP: Dict[str, str] = {"s": "^", "f": "_", "n": "="}
    GAMMES: Dict[str, str] = {
        "0": "C", "1s": "G", "2s": "D", "3s": "A", "4s": "E", "5s": "B",
        "6s": "F#", "7s": "C#", "1f": "F", "2f": "Bb", "3f": "Eb",
        "4f": "Ab", "5f": "Db", "6f": "Gb", "7f": "Cb"
    }
    

    def __init__(self, file_name: Optional[str] = None, content: Optional[str] = None):
        """
        Initialize the converter with either a file name or content.
        
        Args:
            file_name: Path to the MEI file
            content: Raw MEI file content
        """
        self.content = content if content else self._read_file(file_name)
        self.measures: List = []
        self.measures_content: Dict[int, List[str]] = {}
        self.abc_content: str = ""
        self.notes_labels: List[str] = []
        self.pause_labels: List[str] = []
        self.score_def = self._find_score_def()
        self._find_measures()

    @staticmethod
    def _read_file(file_name: str) -> str:
        """Read file content with proper encoding."""
        with open(file_name, "r", encoding="utf-8") as f:
            return f.read()

    def _get_measures_labels(self) -> None:
        """Extract labels for each measure and store them."""
        self.measures_content = {}
        self.notes_labels = []
        self.pause_labels = []

        for i, measure in enumerate(self.measures):
            measure_notes = []
            measure_content = self._extract_measure_content(measure)

            for element in measure_content:
                if self._is_beam(element):
                    beam_notes = self._get_beam_notes(element)
                    beam_notes_labels = [self._parse_note(n) for n in beam_notes]
                    measure_notes.append("".join(beam_notes_labels))
                    self.notes_labels.extend(beam_notes_labels)
                elif self._is_note(element):
                    note_label = self._parse_note(element)
                    measure_notes.append(note_label)
                    self.notes_labels.append(note_label)
                elif self._is_rest(element):
                    rest_label = self._parse_rest(element)
                    measure_notes.append(rest_label)
                    self.pause_labels.append(rest_label)

            self.measures_content[i] = measure_notes

    def mei_to_abc(self) -> str:
        """
        Convert the MEI file content to ABC notation.
        
        Returns:
            ABC notation string
        """
        abc_content = [
            "X:1",
            f'M:{self.score_def.meter_count}/{self.score_def.meter_unit}',
            f'K:{self.score_def.key}',
            "L:1/16",
            f'K: clef={self.score_def.clef}'
        ]

        self._get_measures_labels()

        measure_lines = [" ".join(notes) + " |" for notes in self.measures_content.values()]
        abc_content.extend(measure_lines)
        abc_content.append("]")

        self.abc_content = "\n".join(abc_content)
        return self.abc_content

    def treble_clef_transposition(self) -> List[str]:
        """
        Convert notes from the current clef to treble clef.
        
        Returns:
            List of notes in treble clef
        """
        return [self._convert_note_to_treble(self.score_def.clef, note) 
                for note in self.notes_labels]

    @classmethod
    def _convert_note_to_treble(cls, clef: str, note: str) -> str:
        """Convert a note from a given clef to its equivalent in the treble clef."""
        if clef not in CLEF_TO_TREBLE:
            return note

        clef_mapping = CLEF_TO_TREBLE[clef]
        sorted_keys = sorted(clef_mapping.keys(), key=len, reverse=True)
        pattern = '|'.join(re.escape(key) for key in sorted_keys)

        return re.sub(pattern, lambda m: clef_mapping.get(m.group(0), m.group(0)), note)


class RegexMEIConverter(BaseMEIConverter):
    """MEI to ABC converter using regex-based parsing."""
    
    # Pre-compiled regex patterns
    MEASURE_PATTERN = re.compile(r"<measure[\s\S]*?measure>")
    NOTE_PATTERN = re.compile(r'pname="([^"]*)"')
    OCTAVE_PATTERN = re.compile(r'oct="([^"]*)"')
    DURATION_PATTERN = re.compile(r'dur="([^"]*)"')
    ACCID_PATTERN = re.compile(r'accid="([^"]*)"')
    SCORE_DEF_PATTERN = re.compile(r"<scoreDef[\s\S]*?scoreDef>")
    KEY_PATTERN = re.compile(r'key.sig="([^"]*)"')
    METER_PATTERN = re.compile(r'meter.count="([^"]*)"')
    UNIT_PATTERN = re.compile(r'meter.unit="([^"]*)"')
    STAFF_DEF_PATTERN = re.compile(r"<staffDef.*?/>")
    CLEF_SHAPE_PATTERN = re.compile(r'clef.shape="([^"]*)"')
    CLEF_LINE_PATTERN = re.compile(r'clef.line="([^"]*)"')

    def _find_measures(self) -> None:
        self.measures = self.MEASURE_PATTERN.findall(self.content)

    @lru_cache(maxsize=128)
    def _extract_measure_content(self, measure: str) -> List[str]:
        return re.findall(
            r"<beam[\s\S]*?beam>|<note.*?/>|<note.[\s\S]*?note>|<rest.*?/>|<multiRest.*?/>",
            measure,
        )

    @lru_cache(maxsize=256)
    def _parse_note(self, note: str) -> str:
        value = self.NOTE_PATTERN.search(note).group(1)
        octave = self.OCTAVES[int(self.OCTAVE_PATTERN.search(note).group(1))]
        duration = self.DURATION_MAPPING[self.DURATION_PATTERN.search(note).group(1)]

        if 'dots="' in note:
            if duration == '/':
                duration = '3/4'
            elif duration == '//':
                duration = '3/8'
            else:
                duration = f"{int(float(duration) * 1.5)}"

        accid_match = self.ACCID_PATTERN.search(note)
        if accid_match:
            value = f"{self.ACCID_MAP.get(accid_match.group(1), '')}{value}"

        return f"{value}{octave}{duration}"

    def _find_score_def(self) -> ScoreDefinition:
        score_def = self.SCORE_DEF_PATTERN.findall(self.content)
        if not score_def:
            return ScoreDefinition()

        key = self.KEY_PATTERN.findall(score_def[0])
        meter_count = self.METER_PATTERN.findall(score_def[0])
        meter_unit = self.UNIT_PATTERN.findall(score_def[0])
        staff_defs = self.STAFF_DEF_PATTERN.findall(score_def[0])

        key = self.GAMMES.get(key[0], "") if key else ""
        meter_count = int(meter_count[0]) if meter_count else 4
        meter_unit = int(meter_unit[0]) if meter_unit else 4

        clef = ""
        if staff_defs:
            clef_shape = self.CLEF_SHAPE_PATTERN.search(staff_defs[0])
            clef_line = self.CLEF_LINE_PATTERN.search(staff_defs[0])
            if clef_shape and clef_line:
                clef = clef_shape.group(1) + clef_line.group(1)

        return ScoreDefinition(
            key=key,
            meter_count=meter_count,
            meter_unit=meter_unit,
            clef=clef
        )

    def _is_beam(self, element: str) -> bool:
        return element.startswith("<beam")

    def _is_note(self, element: str) -> bool:
        return element.startswith("<note")

    def _is_rest(self, element: str) -> bool:
        return element.startswith("<rest") or element.startswith("<multiRest")

    def _get_beam_notes(self, beam: str) -> List[str]:
        return re.findall(r"<note.*?/>|<note.[\s\S]*?note>", beam)

    def _parse_rest(self, rest: str) -> str:
        if rest.startswith("<multiRest"):
            duration = re.search(r'num="([^"]*)"', rest).group(1)
            return f"Z{duration}"
        else:
            duration = self.DURATION_MAPPING[self.DURATION_PATTERN.search(rest).group(1)]
            return f"z{duration}"


class XMLMEIConverter(BaseMEIConverter):
    """MEI to ABC converter using XML-based parsing."""

    def __init__(self, file_name: Optional[str] = None, content: Optional[str] = None):

        self.content = content if content else self._read_file(file_name)
        
        try:
            self.root = ET.fromstring(self.content)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML content: {str(e)}")
            
        super().__init__(content=self.content)

    def _find_measures(self) -> None:
        self.measures = self.root.findall(".//measure")

    @lru_cache(maxsize=128)
    def _extract_measure_content(self, measure: ET.Element) -> List[ET.Element]:
        elements = []
        elements.extend(measure.findall("beam"))
        elements.extend(measure.findall("note"))
        elements.extend(measure.findall("rest"))
        elements.extend(measure.findall("multiRest"))
        return elements

    @lru_cache(maxsize=256)
    def _parse_note(self, note: ET.Element) -> str:
        value = note.get("pname", "")
        octave = self.OCTAVES[int(note.get("oct", "4"))]
        duration = self.DURATION_MAPPING[note.get("dur", "4")]

        if note.get("dots") is not None:
            if duration == '/':
                duration = '3/4'
            elif duration == '//':
                duration = '3/8'
            else:
                duration = f"{int(float(duration) * 1.5)}"

        accid = note.get("accid")
        if accid:
            value = f"{self.ACCID_MAP.get(accid, '')}{value}"

        return f"{value}{octave}{duration}"

    def _find_score_def(self) -> ScoreDefinition:
        score_def = self.root.find(".//scoreDef")
        if score_def is None:
            return ScoreDefinition()

        key_sig = score_def.find(".//keySig")
        key = self.GAMMES.get(key_sig.get("sig", ""), "") if key_sig is not None else ""

        meter = score_def.find(".//meterSig")
        meter_count = int(meter.get("count", "4")) if meter is not None else 4
        meter_unit = int(meter.get("unit", "4")) if meter is not None else 4

        staff_def = score_def.find(".//staffDef")
        clef = ""
        if staff_def is not None:
            clef_shape = staff_def.get("clef.shape", "")
            clef_line = staff_def.get("clef.line", "")
            clef = clef_shape + clef_line

        return ScoreDefinition(
            key=key,
            meter_count=meter_count,
            meter_unit=meter_unit,
            clef=clef
        )

    def _is_beam(self, element: ET.Element) -> bool:
        return element.tag == "beam"

    def _is_note(self, element: ET.Element) -> bool:
        return element.tag == "note"

    def _is_rest(self, element: ET.Element) -> bool:
        return element.tag in ("rest", "multiRest")

    def _get_beam_notes(self, beam: ET.Element) -> List[ET.Element]:
        return beam.findall("note")

    def _parse_rest(self, rest: ET.Element) -> str:
        if rest.tag == "multiRest":
            duration = rest.get("num", "1")
            return f"Z{duration}"
        else:
            duration = self.DURATION_MAPPING[rest.get("dur", "4")]
            return f"z{duration}"


def convert_zip(zip_path: str, number_of_files: int = -1, max_workers: int = 4, 
               converter_class=XMLMEIConverter) -> List[BaseMEIConverter]:
    """
    Convert all MEI files in a ZIP archive to ABC notation using parallel processing.
    
    Args:
        zip_path: Path to the ZIP archive
        number_of_files: Number of files to process (-1 for all)
        max_workers: Maximum number of worker threads
        converter_class: The converter class to use (XMLMEIConverter by default)
        
    Returns:
        List of converter instances
    """
    converters = []
    with ZipFile(zip_path, "r") as myzip:
        mei_files = [f for f in myzip.namelist() 
                    if f.startswith("labels/") and f.endswith(".mei")][:number_of_files]
        
        def process_file(mei_file: str) -> BaseMEIConverter:
            with myzip.open(mei_file) as f:
                file_content = f.read().decode("utf-8")
                return converter_class(content=file_content)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            converters = list(executor.map(process_file, mei_files))

    return converters 

def process_file_with_converter(file_path: str, converter_class) -> Tuple[float, bool]:
    """
    Process a single file with a given converter class and return processing time and success status.
    
    Args:
        file_path: Path to the MEI file
        converter_class: The converter class to use
        
    Returns:
        Tuple of (processing_time, success_status)
    """
    try:
        start_time = time.time()
        converter = converter_class(file_name=file_path)
        converter.mei_to_abc()
        end_time = time.time()
        return end_time - start_time, True
    except Exception as e:
        print(f"Error processing {file_path} with {converter_class.__name__}: {str(e)}")
        return 0.0, False

def compare_converters(folder_path: str, converter_classes: List[BaseMEIConverter]) -> pd.DataFrame:
    """
    Compare the performance of different MEI converters on a folder of files.
    
    Args:
        folder_path: Path to the folder containing MEI files
        num_files: Number of files to process for each converter
        
    Returns:
        DataFrame with performance metrics
    """
    results = []
    
    mei_files = os.listdir(folder_path)
    
    if not mei_files:
        raise ValueError(f"No MEI files found in {folder_path}")

    print(f"Found {len(mei_files)} MEI files to process")
    
    for converter_class in converter_classes:
        print(f"\nTesting {converter_class.__name__}...")
        converter_results = []
        
        for file in tqdm(mei_files):
            file_path = os.path.join(folder_path, file)
            processing_time, success = process_file_with_converter(file_path, converter_class)
            converter_results.append({
                'file': file,
                'processing_time': processing_time,
                'success': success
            })
        
        # Statistics
        successful_runs = [r for r in converter_results if r['success']]
        if successful_runs:
            avg_time = sum(r['processing_time'] for r in successful_runs) / len(successful_runs)
            min_time = min(r['processing_time'] for r in successful_runs)
            max_time = max(r['processing_time'] for r in successful_runs)
            median_time = pd.Series([r['processing_time'] for r in successful_runs]).median()
            success_rate = len(successful_runs) / len(converter_results) * 100
        else:
            avg_time = min_time = max_time = 0
            success_rate = 0
            
        results.append({
            'converter': converter_class.__name__,
            'avg_time': avg_time,
            'median_time': median_time,
            'min_time': min_time,
            'max_time': max_time,
            'success_rate': success_rate,
            'total_files': len(converter_results),
            'successful_files': len(successful_runs)
        })
    
    return pd.DataFrame(results)


def convert_to_midi(detections, output_midi: str = "output.mid"):
    """
    Simulated function to convert detections to MIDI.
    In a real scenario, this should convert the list of note detections into a MIDI file.
    """
    print(f"[INFO] Simulating MIDI conversion...")
    with open(output_midi, "w") as f:
        f.write("MIDI simulation for detections.\n")
        f.write(f"Detected {len(detections)} staves\n")
        for i, det in enumerate(detections):
            f.write(f"Staffline {i}: {len(det[0].boxes) if det else 0} detections\n")
    print(f"[INFO] MIDI file written to {output_midi}")
