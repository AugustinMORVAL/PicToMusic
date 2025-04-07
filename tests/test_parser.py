import pytest
import numpy as np
import cv2
import os
from sonatabene.parser import PParser
from sonatabene.scoretyping import StaffLine, Note

@pytest.fixture
def parser():
    return PParser()

@pytest.fixture
def sample_image():
    sample_path = "resources/samples/drum.jpg"
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample image not found at {sample_path}")
    return cv2.imread(sample_path)

# Core Image Processing Tests
def test_image_loading_and_processing(parser, sample_image):
    # Test loading from array
    loaded_image = parser.load_image(sample_image)
    assert isinstance(loaded_image, np.ndarray)
    assert loaded_image.shape == sample_image.shape[:2]
    assert parser.original_image is not None
    assert parser.image is not None
    assert parser.processed_image is not None
    
    # Test loading from file
    test_path = "resources/samples/drum.jpg"
    loaded_from_file = parser.load_image(test_path)
    assert isinstance(loaded_from_file, np.ndarray)
    
    # Test nonexistent file
    with pytest.raises(FileNotFoundError):
        parser.load_image("nonexistent.png")
        
    # Test resizing
    resized = parser.resize(sample_image, max_dim=300)
    assert isinstance(resized, np.ndarray)
    assert max(resized.shape) <= 300
    assert parser.original_shape == sample_image.shape
    assert parser.resized_shape is not None

# Staff Line Detection Tests
def test_staff_line_detection(parser, sample_image):
    parser.load_image(sample_image)
    
    # Test staff line detection with default parameters
    staff_lines = parser.find_staff_lines()
    assert isinstance(staff_lines, list)
    assert len(staff_lines) > 0
    assert all(isinstance(line, StaffLine) for line in staff_lines)
    assert all(line.notes == [] for line in staff_lines)
    
    # Test with custom parameters
    staff_lines_custom = parser.find_staff_lines(
        dilate_iterations=5,
        min_contour_area=5000,
        pad_size=10
    )
    assert isinstance(staff_lines_custom, list)
    assert len(staff_lines_custom) > 0

# Note Detection Tests
def test_note_detection(parser, sample_image):
    parser.load_image(sample_image)
    staff_lines = parser.find_staff_lines()
    
    # Test note detection with default parameters
    staff_lines_with_notes = parser.find_notes(staff_lines)
    assert isinstance(staff_lines_with_notes, list)
    assert len(staff_lines_with_notes) > 0
    assert all(len(line.notes) > 0 for line in staff_lines_with_notes)
    assert all(isinstance(note, Note) for line in staff_lines_with_notes for note in line.notes)
    
    # Test with custom parameters
    staff_lines_with_notes_custom = parser.find_notes(
        staff_lines,
        dilate_iterations=3,
        min_contour_area=100,
        pad_size=5,
        max_horizontal_distance=5,
        overlap_threshold=0.3
    )
    assert isinstance(staff_lines_with_notes_custom, list)
    assert len(staff_lines_with_notes_custom) > 0

# Visualization Tests
def test_visualization(parser, sample_image):
    parser.load_image(sample_image)
    staff_lines = parser.find_staff_lines()
    staff_lines = parser.find_notes(staff_lines)
    contours = parser.find_contours(parser.processed_image)
    
    # Test contour drawing with midpoints
    drawn_contours = parser.draw_contours(parser.original_image, contours, show_midpoints=True)
    assert isinstance(drawn_contours, np.ndarray)
    assert drawn_contours.shape == parser.original_image.shape
    
    # Test staff line drawing with different options
    drawn_staff = parser.draw_staff_lines(
        parser.original_image,
        staff_lines,
        show_staff_bounds=False,
        show_staff_contours=True,
        show_note_bounds=True,
        show_note_contours=False
    )
    assert isinstance(drawn_staff, np.ndarray)
    assert drawn_staff.shape == parser.original_image.shape

def test_find_staff_lines(parser, sample_image):
    parser.load_image(sample_image)
    staff_lines = parser.find_staff_lines()
    assert isinstance(staff_lines, list)
    assert len(staff_lines) > 0
    assert all(isinstance(line, StaffLine) for line in staff_lines)
    assert all(line.notes == [] for line in staff_lines)
    assert all(line.image is not None for line in staff_lines)
    assert all(line.contour is not None for line in staff_lines)
    assert all(line.bounds is not None for line in staff_lines)

def test_find_notes(parser, sample_image):
    parser.load_image(sample_image)
    staff_lines = parser.find_staff_lines()
    staff_lines_with_notes = parser.find_notes(staff_lines)
    
    assert isinstance(staff_lines_with_notes, list)
    assert len(staff_lines_with_notes) > 0
    assert all(isinstance(line, StaffLine) for line in staff_lines_with_notes)
    assert all(len(line.notes) > 0 for line in staff_lines_with_notes)
    assert all(isinstance(note, Note) for line in staff_lines_with_notes for note in line.notes)
    assert all(note.image is not None for line in staff_lines_with_notes for note in line.notes)
    assert all(note.contour is not None for line in staff_lines_with_notes for note in line.notes)
    assert all(note.bounds is not None for line in staff_lines_with_notes for note in line.notes)

def test_group_note_components(parser, sample_image):
    parser.load_image(sample_image)
    contours = parser.find_contours(parser.processed_image)
    grouped_contours = parser.group_note_components(contours)
    assert isinstance(grouped_contours, list)
    assert all(isinstance(cnt, np.ndarray) for cnt in grouped_contours)

def test_remove_staff_lines(parser, sample_image):
    parser.load_image(sample_image)
    cleaned = parser.remove_staff_lines(parser.processed_image)
    assert isinstance(cleaned, np.ndarray)
    assert cleaned.shape == parser.processed_image.shape

def test_extract_element(parser, sample_image):
    parser.load_image(sample_image)
    bounds = (0, 0, 100, 100)
    extracted = parser.extract_element(bounds)
    assert isinstance(extracted, np.ndarray)
    assert extracted.shape == (bounds[3], bounds[2])

def test_add_padding(parser, sample_image):
    padded = parser._add_padding(sample_image, pad_size=10)
    assert isinstance(padded, np.ndarray)
    assert padded.shape[0] == sample_image.shape[0] + 20
    assert padded.shape[1] == sample_image.shape[1] + 20 

def test_extract_contours(parser, sample_image):
    parser.load_image(sample_image)
    contours = parser.find_contours(parser.processed_image)
    
    # Test horizontal extraction
    horizontal_regions = parser.extract_contours(sample_image, contours, axis=0)
    assert isinstance(horizontal_regions, list)
    assert len(horizontal_regions) > 0
    assert all(isinstance(region, np.ndarray) for region in horizontal_regions)
    
    # Test vertical extraction
    vertical_regions = parser.extract_contours(sample_image, contours, axis=1)
    assert isinstance(vertical_regions, list)
    assert len(vertical_regions) > 0
    assert all(isinstance(region, np.ndarray) for region in vertical_regions)
    
    # Test full height extraction
    full_height_regions = parser.extract_contours(sample_image, contours, axis=0, full_height=True)
    assert isinstance(full_height_regions, list)
    assert len(full_height_regions) > 0
    assert all(region.shape[0] == sample_image.shape[0] for region in full_height_regions)
    
    # Test invalid axis
    with pytest.raises(ValueError):
        parser.extract_contours(sample_image, contours, axis=2)

def test_merge_group(parser, sample_image):
    parser.load_image(sample_image)
    contours = parser.find_contours(parser.processed_image)
    
    test_group = []
    for i in range(3):
        x, y, w, h = cv2.boundingRect(contours[i])
        test_group.append((contours[i], (x, y, w, h)))
    
    merged = parser._PParser__merge_group(test_group)
    assert isinstance(merged, np.ndarray)
    assert merged.shape[0] == 4  
    assert merged.shape[1] == 1
    assert merged.shape[2] == 2  

# Utility Tests
def test_image_utils(parser, sample_image):
    parser.load_image(sample_image)
    
    # Test padding
    padded = parser._add_padding(sample_image, pad_size=10)
    assert isinstance(padded, np.ndarray)
    assert padded.shape[0] == sample_image.shape[0] + 20
    assert padded.shape[1] == sample_image.shape[1] + 20
    
    # Test element extraction
    bounds = (0, 0, 100, 100)
    extracted = parser.extract_element(bounds)
    assert isinstance(extracted, np.ndarray)
    assert extracted.shape == (bounds[3], bounds[2])
    
    # Test contour extraction
    contours = parser.find_contours(parser.processed_image)
    
    # Horizontal extraction
    horizontal_regions = parser.extract_contours(sample_image, contours, axis=0)
    assert isinstance(horizontal_regions, list)
    assert len(horizontal_regions) > 0
    
    # Vertical extraction
    vertical_regions = parser.extract_contours(sample_image, contours, axis=1)
    assert isinstance(vertical_regions, list)
    assert len(vertical_regions) > 0
    
    # Test invalid axis
    with pytest.raises(ValueError):
        parser.extract_contours(sample_image, contours, axis=2) 