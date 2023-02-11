""" Driver code."""
from utils.demonstration import annotate_video
from data.encoding.landmark_encoder import encode_video_with_metadata
from data.preprocessing.annotater import annotate_videos

# annotate_videos(data_source='', save_directory='')

# hands, poses, metadata = encode_video_with_metadata('https://www.youtube.com/watch?v=uVjKieHqD_M', device='cpu')

# annotate_video('https://www.youtube.com/watch?v=uVjKieHqD_M')

# print(hands.size())
# print(poses.size())
# print(metadata)
