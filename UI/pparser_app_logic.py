from sonatabene.parser import PParser	
import numpy as np
import streamlit as st

def parse_music_sheet(image, progress_bar, params=None):
    parser = PParser()
    total_progress = 0
    progress_bar.progress(0)
    
    if params is None:
        params = {
            # 'resize_max_dim': 1600,
            'staff_dilate_iterations': 3,
            'staff_min_contour_area': 10000,
            'staff_pad_size': 0,
            'note_dilate_iterations': 3,
            'note_min_contour_area': 75,
            'note_pad_size': 0,
            'max_horizontal_distance': 10,
            'overlap_threshold': 0.2
        }
    
    image = parser.load_image(image)
    # image = parser.resize(image, max_dim=params['resize_max_dim'])
    
    staff_lines = parser.find_staff_lines(
        dilate_iterations=params['staff_dilate_iterations'],
        min_contour_area=params['staff_min_contour_area'],
        pad_size=params['staff_pad_size']
    )
    
    total_steps = len(staff_lines) + 1
    progress_increment = 100 / total_steps
    total_progress += progress_increment
    progress_bar.progress(int(total_progress))
    
    staff_lines = parser.find_notes(
        staff_lines,
        dilate_iterations=params['note_dilate_iterations'],
        min_contour_area=params['note_min_contour_area'],
        pad_size=params['note_pad_size'],
        max_horizontal_distance=params['max_horizontal_distance'],
        overlap_threshold=params['overlap_threshold']
    )
    
    staff_visualization = parser.draw_staff_lines(
        image.copy(), 
        staff_lines,
        show_staff_bounds=False,
        show_staff_contours=True,
        show_note_bounds=False,
        show_note_contours=False,
    )
    
    notes_visualization = parser.draw_staff_lines(
        image.copy(),
        staff_lines,
        show_staff_bounds=False,
        show_staff_contours=False,
        show_note_bounds=True,
        show_note_contours=True
    )
    
    # Extract note information
    all_notes = []
    for staff in staff_lines:
        staff_notes = []
        for note in staff.notes:
            staff_notes.append({
                'global_index': note.index,
                'relative_index': note.relative_index,
                'line_index': note.line_index,
                'position': note.absolute_position
            })
        all_notes.append(staff_notes)
    
    progress_bar.progress(100)
    
    return staff_lines, staff_visualization, notes_visualization, all_notes