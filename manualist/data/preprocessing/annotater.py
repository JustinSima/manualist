""" Driver function for preprocessing.
Loads videos, labels with landmarks, saves landmarks and metadata.
"""
import os
import json

import torch
from tqdm import tqdm

import data.encoding.landmark_encoder as landmark_encoder


def annotate_videos(data_source, save_directory):
    logged_errors = {}

    with open(data_source, 'r') as json_file:
        dataset = json.load(json_file)

    for class_dict in tqdm(dataset):
        for sample_dict in tqdm(class_dict['instances']):
            # Create a sample and populate.

            label = class_dict['gloss']
            split = sample_dict['split']
            url = sample_dict['url']
            vid_id = sample_dict['video_id']

            # Check if file exists before attempting to create.
            sample_dir = os.path.join(save_directory, split, vid_id)
            if os.path.exists(sample_dir):
                print('Sample already created.')
                continue

            # Create sample.
            hands, poses, metadata = landmark_encoder.encode_video_with_metadata(video_path=url)
            
            # Log errors.
            if hands is None:
                logged_errors[vid_id] = label
            
            # Save succesful attempts.
            else:
                metadata['label'] = label
                metadata['split'] = split
                metadata['url'] = url
                metadata['vid_id'] = vid_id
                
                os.mkdir(sample_dir)
                with open(sample_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f)
                    
                torch.save(hands, sample_dir / 'hands.pt')
                torch.save(poses, sample_dir / 'pose.pt')

    if len(logged_errors) > 0:
        log_path = os.path.join(save_directory, 'log', 'error_log.json')
        with open(log_path, 'w') as f:
            json.dump(logged_errors, f)
