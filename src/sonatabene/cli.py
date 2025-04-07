import click
import yaml
from sonatabene import __version__

@click.group(name='snb', help='A command line tool to convert images to music')
@click.version_option(__version__)
def snb():
    pass

@snb.group(name='model', help='Set of commands to launch training, prediction or evaluation models')
def model():
    pass

@model.command(name='train', help='Train a YOLO model on a custom dataset')
@click.option('--data-path', '-d', required=True, help='Path to dataset configuration file (e.g. data/dataset.yaml)')
@click.option('--model-path', '-m', default='yolo11n.pt', help='Path to initial model weights')
@click.option('--config-path', '-c', default='configs/training_config.yaml', help='Path to training configuration YAML file')
def train(data_path: str, model_path: str, config_path: str):
    """Execute the YOLO model training workflow with specified parameters."""
    from sonatabene.model import train
    
    with open(config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    train(
        data_path=data_path,
        model_path=model_path,
        **training_config
    )

@model.command(name='predict', help='Predict notes from an image')
@click.option('--image-path', '-i', required=True, help='Path to the image file for YOLO predictions')
@click.option('--model-path', '-m', default='models/chopin.pt', help='Path to initial model weights')
@click.option('--config-path', '-c', default='configs/predict_config.yaml', help='Path to prediction configuration YAML file')
def predict(image_path: str, model_path: str, config_path: str):
    """Execute the YOLO model prediction workflow with specified parameters."""
    from sonatabene.model import predict
    
    with open(config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    predict(
        image_path=image_path,
        model_path=model_path,
        **training_config
    )

@snb.group(name='music', help='Set of commands to convert into music formats')
def music():
    pass

@music.command(name='play', help='Play a MIDI file from YOLO predictions based on image')
@click.option('--image-path', '-i', required=True, help='Path to the image file for YOLO predictions')
@click.option('--model-path', '-m', default='models/chopin.pt', help='Path to the trained YOLO model')
@click.option('--instrument', '-inst', default='Piano', help='Instrument to use for the MIDI output (e.g., PanFlute, Piano, Violin)')
@click.option('--tempo', '-t', default=120, type=int, help='Tempo for the MIDI output in beats per minute (default: 120)')
@click.option('--dynamics', '-d', type=str, help='Dynamic markings in JSON format (e.g., {"p": 40, "f": 100})')
@click.option('--articulation', '-a', type=str, help='Articulation settings in JSON format (e.g., {"staccato": 0.5, "tenuto": 1.0})')
@click.option('--output-format', '-f', type=click.Choice(['midi', 'musicxml', 'pdf', 'wav', 'mp3']), default='midi', help='Output format')
@click.option('--output-file', '-o', help='Path to save the output file')
def play_midi_from_yolo(image_path: str, model_path: str, instrument: str, tempo: int, 
                       dynamics: str, articulation: str, output_format: str, output_file: str):
    """Generate MIDI from YOLO predictions and play it."""
    from sonatabene.model import predict
    from sonatabene.converter.converter_abc import abc_to_midi, abc_to_musicxml, abc_to_pdf, abc_to_audio
    from sonatabene.converter.converter_yolo import yolo_to_abc
    import json
    import loguru
    from sonatabene.parser import PParser
    import cv2

    dynamics_dict = json.loads(dynamics) if dynamics else None
    articulation_dict = json.loads(articulation) if articulation else None

    loguru.logger.info("Parsing stafflines...")
    parser = PParser()
    parser.load_image(image_path)
    stafflines = parser.find_staff_lines(min_contour_area=10000)

    staffs = [cv2.cvtColor(staffline.image, cv2.COLOR_RGB2BGR) for staffline in stafflines]
    
    predictions = []
    loguru.logger.info("Predicting Notes...")
    for i, staff in enumerate(staffs):
        result = predict(image=staff, model_path=model_path)[0]
        loguru.logger.info(f"Predicted {i}/{len(staffs)}")
        predictions.append(result)
    
    loguru.logger.info("Converting results to ABC...")
    abc = yolo_to_abc(predictions)
    loguru.logger.info(f"ABC formatted : \n{abc}")

    if output_format == 'midi':
        loguru.logger.info("Creating MIDI...")
        abc_to_midi(
            abc, 
            output_file=output_file,
            play=True,
            instrument=instrument,
            tempo_bpm=tempo,
            dynamics=dynamics_dict,
            articulation=articulation_dict
        )
    elif output_format == 'musicxml':
        loguru.logger.info("Creating MusicXML...")
        abc_to_musicxml(abc, output_file)
    elif output_format == 'pdf':
        loguru.logger.info("Creating PDF...")
        abc_to_pdf(abc, output_file)
    elif output_format in ['wav', 'mp3']:
        loguru.logger.info("Creating Audio...")
        abc_to_audio(abc, output_file, format=output_format)