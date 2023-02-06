""" Define constants."""
import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# --- File paths and directory locations.
DATA_SOURCE = os.path.join(ROOT_DIR, 'data', 'WLASL', 'start_kit', 'WLASL_v0.3.json')
DATA_DIRECTORY = os.path.join(ROOT_DIR, 'data')
LABEL_DIRECTORY = os.path.join(DATA_DIRECTORY, 'labels')

# --- Directories for raw video. Only required until landmarks are created.
# RAW_VIDEO_DIR = os.path.join(ROOT_DIR, 'files', 'WLASL', 'videos')
# VIDEO_DIR = os.path.join(ROOT_DIR, 'files', 'WLASL', 'labeled_videos')
