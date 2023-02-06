import os
import json
from tqdm import tqdm
from dataclasses import dataclass

import utils.landmarks as landmarks


@dataclass
class Sample:
    url: str
    label: str
    split: str
    vid_id: str
    features: dict

    def __init__(self):
        self.features = {}

    def save_sample(self, save_path):
        # Check that features are not None.
        if not self.features:
            print(f'Failed to create features for sample {self.vid_id}.')
            return self.vid_id

        # Handling case where sample is empty.
        if len(self.features) == 1:
            print(f'Features created incorrectly for sample {self.vid_id}.')
            return self.vid_id

        # Prepare output, adjust as needed.
        output_dict = {
            'vid_id':self.vid_id,
            'label':self.label,
            'features':self.features
        }

        with open(save_path, 'w') as save_file:
            json.dump(output_dict, save_file)
        print('Sample succesfully annotated.')

def annotate_videos(data_source, save_directory):
    logged_errors = {}

    with open(data_source, 'r') as json_file:
        dataset = json.load(json_file)

    for class_dict in tqdm(dataset):
        for sample_dict in tqdm(class_dict['instances']):
            # Create a sample and populate.
            sample = Sample()

            sample.label = class_dict['gloss']
            sample.split = sample_dict['split']
            sample.url = sample_dict['url']
            sample.vid_id = sample_dict['video_id']

            # Check if file exists before attempting to create.
            save_name = os.path.join(save_directory, sample.split, sample.vid_id + '.json')
            if os.path.exists(save_name):
                print('Sample already created.')
                continue

            # Create and save sample. Catch and store errors.
            sample.features = landmarks.create_sample(sample.url)
            failure = sample.save_sample(save_path=save_name)

            if failure:
                logged_errors[sample.vid_id] = sample.label

    if len(logged_errors) > 1:
        log_path = os.path.join(data_source, 'log', 'error_log.json')
        with open(log_path, 'w') as f:
            json.dump(logged_errors, f)
