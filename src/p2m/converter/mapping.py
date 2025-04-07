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

CLEF_ABC_MAPPING = {
    "C1": "soprano",         # C clef on the 1st line
    "C2": "mezzosoprano",    # C clef on the 2nd line
    "C3": "alto",            # C clef on the 3rd line
    "C4": "tenor",           # C clef on the 4th line
    "G2": "treble",          # G clef on the 2nd line (standard treble)
    "F3": "baritone-f",      # F clef on the 3rd line
    "F4": "bass"             # F clef on the 4th line (standard bass)
}

GAMMES = {
    "0": "C", "1s": "G", "2s": "D", "3s": "A", "4s": "E", "5s": "B",
    "6s": "F#", "7s": "C#", "1f": "F", "2f": "Bb", "3f": "Eb",
    "4f": "Ab", "5f": "Db", "6f": "Gb", "7f": "Cb"
}