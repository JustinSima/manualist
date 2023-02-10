""" Encode a video as a pytorch Tensor of landmarks."""
import cv2
import mediapipe as mp
import numpy as np
import pafy
import pickle
import torch

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def encode_video(video_path: str, device: str='cpu') -> torch.Tensor:
    """ Loads a video and returns a PyTorch tensor of landmark encodings.
    If video cannot be loaded, returns None.

    Args:
        video_path (str): A path to a local file or YouTube link.
        
    Returns:
        torch.Tensor: PyTorch tensor of landmarks. Returns None if video fails to load.
    """
    hands = []
    poses = []
    
    video_file = load_video(video_path)
    if video_file is None:
        return None
    
    capture = cv2.VideoCapture(video_file)
    # Open hand and pose context manager.
    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as hand_capture, \
    mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as pose_capture:
        while(True):
            success, image = capture.read()
            if not success:
                break
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hand_results = hand_capture.process(image)
            pose_results = pose_capture.process(image)
        
            # Convert MediaPipe results to a tensor if hand is found.
            if hand_results.multi_hand_landmarks:   
                frame_hands = [
                    [p.x, p.y, p.z] for single_hand in hand_results.multi_hand_landmarks 
                    for p in single_hand.landmark
                ]
                
                if len(frame_hands) == 21:
                    frame_hands = [*frame_hands, *[[0., 0., 0.] for _ in range(21)]]
 
                if len(frame_hands) == 42:
                    hands.append(frame_hands)
                    
                frame_pose = [[p.x, p.y, p.z] for p in pose_results.pose_landmarks.landmark[0:15]]
                poses.append(frame_pose)
        
        capture.release()
    
    hands = np.asarray(hands)
    poses = np.asarray(poses)
    hands = hands.reshape(-1, 42, 3)
    poses = poses.reshape(-1, 15, 3)
    hands = torch.from_numpy(hands).float().to(torch.device(device))
    poses = torch.from_numpy(poses).float().to(torch.device(device))
    
    return hands, poses

def load_video(video: str):
    """ Loads video into acceptable format for OpenCV.
    If video is a link to a YouTube video,
        it is loaded and returned as an mp4.
    Otherwise, it is assumed to be a path to a video file,
        and is returned as is.

    Args:
        video (str): A path to a local file or YouTube link.
    """
    # Return if not a YouTube video.
    if 'youtube' not in video:
        return video

    try:
        video = pafy.new(video)
        return video.getbest(preftype='mp4').url
    except OSError:
        return None
