from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
@dataclass
class StaffLine:
    """Represents a staff line and its associated notes in a music score."""
    index: int
    filename: str
    image: np.ndarray
    contour: np.ndarray 
    bounds: Tuple[int, int, int, int]
    notes: List['Note'] = field(default_factory=list)
    
    def show(self) -> str:
        """Return a string representation of the staff line."""
        plt.figure(figsize=(10, 4))
        plt.imshow(self.image, cmap='gray')
        plt.axis('off')
        plt.show()
        return f"StaffLine(index={self.index}, filename={self.filename}, notes={len(self.notes)})"

    def get_notes_with_label(self, label: str) -> List['Note']:
        """Get all notes with a specific label.
        
        Args:
            label: The label to search for
            
        Returns:
            List of notes with the specified label
        """
        return [note for note in self.notes if note.label == label]

@dataclass
class Note:
    """Represents a musical element in the score."""
    index: int
    relative_index: int
    line_index: int
    image: np.ndarray  
    contour: np.ndarray 
    bounds: Tuple[int, int, int, int] 
    full_height_bounds: Tuple[int, int, int, int]
    relative_position: Tuple[int, int] 
    absolute_position: Tuple[int, int] 
    label: Optional[str] = None

    def show(self) -> str:
        """Return a string representation of the note."""
        plt.figure(figsize=(2, 2))
        plt.imshow(self.image, cmap='gray')
        plt.axis('off')
        plt.show()
        return f"Note(index={self.index}, shape={self.image.shape}, label={self.label})"

    def set_label(self, label: str) -> None:
        """Set the label for this note."""
        self.label = label
@dataclass
class Key:
    """Represents a key signature in a music score."""
    line_index: int
    contour: np.ndarray
    bounds: Tuple[int, int, int, int]
    relative_position: Tuple[int, int] 
    absolute_position: Tuple[int, int] 
    metric: Tuple[int, int] = (4, 4)
    label: Optional[str] = None
    gamme: Optional[str] = None