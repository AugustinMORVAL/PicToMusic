from zipfile import ZipFile
from ultralytics import YOLO
from p2m.converter import XMLMEIConverter
import numpy as np
import os
import yaml
import shutil
from PIL import Image
from tqdm import tqdm
import io


MODEL_PATH = "models/bach.pt"
ZIP_PATH = "data/dataset.zip"
YAML_PATH = "data/dataset.yaml"
OUTPUT_PATH = "data/output/"

def check_count(count_names, mei_notes_count, mei_pause_count, mei_score_def):
    expected_counts = np.array([1, mei_notes_count, mei_pause_count, 1, 1])
    actual_counts = np.array([
        count_names.get('clef', 0), 
        count_names.get('note', 0), 
        count_names.get('pause', 0), 
        count_names.get('gamme', 0), 
        count_names.get('metrics', 0)
    ])
    
    if np.array_equal(actual_counts, expected_counts):
        return 'Mapping correct'
    
    return 'Mapping incorrect'

def sort_boxes(boxes):
    boxes_array = boxes.xywhn.cpu().numpy()

    class_indices = boxes.cls.cpu().numpy().reshape(-1, 1)  

    # Concatenate class indices with bounding boxes → (N, 5) array: [class, x_center, y_center, width, height]
    boxes_with_labels = np.hstack((class_indices, boxes_array))
    
    sorted_indices = boxes_with_labels[:, 1].argsort()
    sorted_boxes = boxes_with_labels[sorted_indices] 

    return sorted_boxes

def get_or_add_class_to_yaml(yaml_path, class_name):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    if 'names' not in data:
        data['names'] = {}
    
    for index, name in data['names'].items():
        if name == class_name:
            return int(index)
        
    new_index = max(map(int, data['names'].keys()), default=-1) + 1
    data['names'][new_index] = class_name
    
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return new_index

def associate_class_labels(sorted_boxes, labels_dict, mei_labels, yaml_path):
    class_labels = []
    i_note = 0
    i_pause = 0
    
    for bbox in sorted_boxes:
        if labels_dict[bbox[0]] in ['clef', 'metrics', 'gamme']:
            class_index = get_or_add_class_to_yaml(yaml_path, mei_labels[labels_dict[bbox[0]]])
            class_labels.append(class_index)
        elif labels_dict[bbox[0]] == 'note':
            class_index = get_or_add_class_to_yaml(yaml_path, mei_labels['note_labels'][i_note])
            class_labels.append(class_index)
            i_note += 1
        elif labels_dict[bbox[0]] == 'pause':
            class_index = get_or_add_class_to_yaml(yaml_path, mei_labels['pause_labels'][i_pause])
            class_labels.append(class_index)
            i_pause += 1
    return class_labels

def compare_mei_to_parser(file_name, myzip, output_dir, yaml_path, model):
    with myzip.open('labels/' + file_name + '.mei') as mei_file, myzip.open('images/' + file_name + '.png') as img_file:
        mei_content = mei_file.read().decode('utf-8')
        mei = XMLMEIConverter(content=mei_content)
        mei.mei_to_abc()
        mei_clef, mei_gamme = mei.score_def.clef, mei.score_def.key
        mei_metrics = "4/4" if mei.score_def.meter_count == '' else str(mei.score_def.meter_count) + '/' + str(mei.score_def.meter_unit)

        img_bytes = img_file.read()
        img = Image.open(io.BytesIO(img_bytes))

        result = model(img, verbose=False)[0]

        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        unique_names, counts = np.unique(names, return_counts=True)

        count_names = dict(zip(unique_names, counts))

        check_count_names = check_count(count_names, len(mei.notes_labels), len(mei.pause_labels), mei.score_def)

        if check_count_names != "Mapping correct":
            return "Mapping incorrect"
    
        os.makedirs(f'{output_dir}', exist_ok=True)
        shutil.copyfileobj(img_file, open(f'{output_dir}' + file_name.split('/')[-1] + '.png', 'wb'))

        if mei_clef != 'G2' or mei_clef != '':
            notes_labels = mei.treble_clef_transposition()
        else:
            notes_labels = mei.notes_labels
            
        sorted_boxes = sort_boxes(result.boxes)
        mei_labels = {'note_labels': notes_labels, 'pause_labels': mei.pause_labels, 'clef': mei_clef, 'gamme': mei_gamme, 'metrics': mei_metrics}
        class_labels = associate_class_labels(sorted_boxes, result.names, mei_labels, yaml_path)

        with open(output_dir + file_name.split('/')[-1] + '.txt', 'w') as f:
            for bbox, label in zip(sorted_boxes, class_labels):
                f.write(f"{label} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")  # YOLO format: class x_min y_min x_max y_max

if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with ZipFile(ZIP_PATH, "r") as myzip:
        zip_files = myzip.namelist()

        image_files = [file for file in zip_files if file.endswith('.png') and 'images/' in file]
        mei_files = {os.path.basename(file).replace('.mei', ''): file for file in zip_files if file.endswith('.mei') and 'labels/' in file}

        for image_file in tqdm(image_files, desc="Processing Images", total=len(image_files), unit='images'):
            image_name = os.path.basename(image_file).replace('.png', '')
            corresponding_mei_file = mei_files.get(image_name)

            if corresponding_mei_file:
                compare_mei_to_parser(image_name, myzip, output_dir=OUTPUT_PATH, yaml_path=YAML_PATH, model=YOLO(MODEL_PATH))
            else:
                print(f"No corresponding MEI file found for {image_file}")