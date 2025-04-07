import numpy as np
import pandas as pd
import cv2
import csv
from pathlib import Path
from typing import List, Tuple
from p2m.scoretyping import StaffLine
import zipfile
import os
from tqdm import tqdm
import loguru
import shutil
import platform

def imreshape(image: np.ndarray, shape: int = 128):
    return cv2.resize(image, (shape, shape))

def generate_detection_csv(staff_lines_list: List[List[StaffLine]], 
                         output_filename: str, include_staff: bool = False) -> None:
    """
    Generate a detection CSV file from multiple lists of staff lines and their notes.
    
    Args:
        staff_lines (List[StaffLine]): List of staff line lists, one for each score
        output_filename (str): Path to output CSV file
        include_staff (bool): Whether to include staff lines in the CSV
    """
     
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        
        for staff_lines in staff_lines_list:
            for staff in staff_lines:
                if include_staff:
                    x, y, w, h = staff.bounds
                    width = staff.image.shape[1]
                    height = staff.image.shape[0]
                    label = "staff"
                    
                    writer.writerow([
                        staff.filename, # filename
                        width,          # width
                        height,         # height
                        label,          # class
                        x,              # xmin
                        y,              # ymin
                        x + w,          # xmax
                        y + h           # ymax
                    ])                
                for note in staff.notes:
                    x, y, w, h = note.full_height_bounds
                    width = note.image.shape[1]
                    height = note.image.shape[0]
                    label = note.label if note.label else "note"
                    
                    writer.writerow([
                        staff.filename, # filename
                        width,          # width
                        height,         # height
                        label,          # class
                        x,              # xmin
                        y,              # ymin
                        x + w,          # xmax
                        y + h           # ymax
                    ])
                    

def extract_dataset(zip_path: str, output_path: str):
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Could not find {zip_path}")
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        loguru.logger.info(f"Extracting {zip_path}")
        total_size = sum(file.file_size for file in zip_ref.filelist)
        loguru.logger.info(f"Total size: {total_size} bytes")
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
            for file in zip_ref.filelist:
                zip_ref.extract(file, output_path)
                pbar.update(file.file_size)
    
    loguru.logger.info(f"Successfully extracted {zip_path}")
    
def split_data(input_path: str, output_path: str, batch_size: int, num_batch: int, random_seed: int = None):
    """
    Split the dataset into multiple batches and save them to the output path.
    
    Args:
        input_path (str): Path to the input dataset
        output_path (str): Path to save the batches
        batch_size (int): Number of files per batch
        num_batch (int): Number of batches to create
        random_seed (int, optional): If provided, randomize the order of files using this seed
    """
    
    os.makedirs(output_path, exist_ok=True)
    
    image_files = sorted(os.listdir(os.path.join(input_path, 'images')))
    mei_files = sorted(os.listdir(os.path.join(input_path, 'labels')))
    
    file_pairs = list(zip(image_files, mei_files))
    
    if random_seed is not None:
        import random
        random.seed(random_seed)
        random.shuffle(file_pairs)
        image_files, mei_files = zip(*file_pairs) 
    
    # loguru.logger.info(f"Found {len(image_files)} images and {len(mei_files)} MEI files")
    
    for batch_num in tqdm(range(num_batch), desc="Processing batches", total=num_batch, unit="batch"):
        batch_folder = os.path.join(output_path, f'batch_{batch_num}')
        os.makedirs(batch_folder, exist_ok=True)
        
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(image_files))
        
        # loguru.logger.info(f"Processing batch {batch_num} ({end_idx - start_idx} files)")
        
        os.makedirs(os.path.join(batch_folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(batch_folder, 'labels'), exist_ok=True)
        
        for file_name, mei_file in zip(image_files[start_idx:end_idx], mei_files[start_idx:end_idx]):
            src_img = os.path.join(input_path, 'images', file_name)
            dst_img = os.path.join(batch_folder, 'images', file_name)
            shutil.copy2(src_img, dst_img)
            # loguru.logger.debug(f"Copied {src_img} to {dst_img}")

            src_mei = os.path.join(input_path, 'labels', mei_file)
            dst_mei = os.path.join(batch_folder, 'labels', mei_file)
            shutil.copy2(src_mei, dst_mei)
            # loguru.logger.debug(f"Copied {src_mei} to {dst_mei}")
                
    loguru.logger.info(f"Successfully processed {num_batch} batches and saved to {output_path}")
    return output_path

def get_musescore_path():
    """
    Detect the operating system and return the correct MuseScore path.
    
    Returns:
        str: Path to the MuseScore executable.
    """
    system = platform.system()

    if system == 'Windows':
        # For Windows, MuseScore is usually in 'Program Files'
        musescore_path = os.path.join(os.environ['ProgramFiles'], 'MuseScore 4', 'bin', 'MuseScore4.exe')
        
        if not os.path.exists(musescore_path):
            raise FileNotFoundError("MuseScore 4 not found on Windows. Please install it.")
    
    elif system == 'Linux':
        # For Linux/WSL, check if we are running inside WSL
        if 'WSL_DISTRO_NAME' in os.environ:
            # Running inside WSL, convert Windows path to WSL path format
            musescore_path = '/mnt/c/Program Files/MuseScore 4/bin/MuseScore4.exe'
        else:
            # Native Linux setup (ensure MuseScore is installed for Linux)
            musescore_path = '/usr/bin/MuseScore4'  # Path may vary based on installation method

        if not os.path.exists(musescore_path):
            raise FileNotFoundError("MuseScore not found on Linux/WSL. Please install it.")

    elif system == 'Darwin':
        # For macOS, MuseScore is usually installed in the Applications folder
        musescore_path = '/Applications/MuseScore 4.app/Contents/MacOS/MuseScore4'
        
        if not os.path.exists(musescore_path):
            raise FileNotFoundError("MuseScore 4 not found on macOS. Please install it.")
    
    else:
        raise OSError(f"Unsupported operating system: {system}")

    return musescore_path