""" Utilize MediaPipe to capture hand and head landmarks of a video."""
import cv2
import mediapipe as mp
import pafy

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def create_head_landmark(video_file):
    video_landmarks = []
    result_container = {}

    # Load video and initialize hands.
    capture = cv2.VideoCapture(video_file)
    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3) as pose:

        # Iterate through frames.
        while (True):
            success, image = capture.read()

            if not success:
                break

            # Change to unwritable for efficiency, read hands.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape
            results = pose.process(image)

            # Store landmarks.
            if results.pose_landmarks:
                x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
                y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height
                video_landmarks.append((x, y))

            else:
                video_landmarks.append((None, None))

    capture.release()

    # Convert results to desired format.
    for ind, _ in enumerate(video_landmarks):
        result_container[ind] = video_landmarks[ind]

    return result_container

def create_hand_landmarks(video_file):
    video_landmarks = []
    result_container = {}

    # Load video and initialize hands.
    capture = cv2.VideoCapture(video_file)
    with mp_hands.Hands(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3) as hands:

        # Iterate through frames.
        while (True):
            success, image = capture.read()

            if not success:
                break

            # Change to unwritable for efficiency, read hands.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Store landmarks.
            if results.multi_hand_landmarks:
                hand_locations = []
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_results = [hand_landmarks.landmark[i] for i in range(21)]
                    hand_locations.append(landmark_results)
                video_landmarks.append(hand_locations)

            else:
                video_landmarks.append([[None for i in range(21)], [None for i in range(21)]])

    capture.release()

    # Convert results to desired format.
    for frame_ind, frame in enumerate(video_landmarks):
        landmark_dict = {}
        try:
            landmark_dict['hand0'] = { lndmrk:(frame[0][lndmrk].x, frame[0][lndmrk].y, frame[0][lndmrk].z) for lndmrk, _ in enumerate(frame[0]) }
            landmark_dict['hand1'] = { lndmrk:(frame[1][lndmrk].x, frame[1][lndmrk].y, frame[1][lndmrk].z) for lndmrk, _ in enumerate(frame[1]) }
        except AttributeError:
            landmark_dict['hand0'] = { None:(None, None, None) for i in range(21) }
            landmark_dict['hand1'] = { None:(None, None, None) for i in range(21) }
        except IndexError:
            # TODO: When a second hand is added, it may be the left, right, etc.
            # Need to standardize left/right.
            landmark_dict['hand1'] = { None:(None, None, None) for i in range(21) }

        result_container[frame_ind] = landmark_dict

    return result_container

def create_landmark_sample(video_file):
    """ Create sample of hand and head landmarks for video file."""
    sample = {}

    video_file = convert_video(video_file)

    hand_landmarks = create_hand_landmarks(video_file=video_file)
    head_landmark = create_head_landmark(video_file=video_file)

    if len(hand_landmarks) != len(head_landmark):
        print('Length mismatch error: hand and head landmarks not the same length.')
        return {}

    for ind in range(len(head_landmark)):
        sample[ind] = {
            'hand0':hand_landmarks[ind]['hand0'],
            'hand1':hand_landmarks[ind]['hand1'],
            'face':head_landmark[ind]
        }

    return sample

def convert_video(video_file):
    """ Attempt to convert YouTube video link to mp4."""
    # Return if nothing needed.
    if 'youtube' not in video_file:
        return video_file

    try:
        video = pafy.new(video_file)
        return video.getbest(preftype='mp4').url
    except OSError:
        return video_file
