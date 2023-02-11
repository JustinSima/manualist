""" Function to load video and display with landmarks."""
import cv2
import mediapipe as mp
import pafy
import torch

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def display_annotated_video(video_path: str) -> torch.Tensor:
    """ Loads a video and displays how that video would be labeled by MediaPipe.

    Args:
        video_path (str): A path to a local file or YouTube link.
    """
    video_file = load_video(video_path)
    
    capture = cv2.VideoCapture(video_file)

    # Open hand and pose context manager.
    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as hands, \
    mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as pose:
        while(True):
            success, image = capture.read()
            if not success:
                break
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(image)
            pose_results = pose.process(image)
        
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
        
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

        capture.release()

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
        return video
