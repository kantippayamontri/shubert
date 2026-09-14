import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
import pickle
import gzip
from datetime import datetime
from pathlib import Path
import decord
import argparse
import json
import csv
import time
from pdb import set_trace as st


def compute_stats_from_pose_data(pose_path, stats_path, video_file):
    landmark_json_path = Path(f"{pose_path}{video_file.split('/')[-1].rsplit('.', 1)[0]}_pose.json")
    stats_json_path = Path(f"{stats_path}{video_file.split('/')[-1].rsplit('.', 1)[0]}_stats.json")
    
    
    # Load the pose data
    with open(landmark_json_path, 'r') as f:
        pose_data = json.load(f)
    
        # Initialize stats and max counts
        stats = {}
        max_counts = {'#face': 0, '#hands': 0, '#pose': 0}
        
        # Compute stats and track max counts
        for frame, landmarks in pose_data.items():
            if landmarks is None:
                presence = {'#face': 0, '#hands': 0, '#pose': 0}
            else:
                presence = {
                    '#face': len(landmarks.get('face_landmarks', [])) if landmarks.get('face_landmarks') else 0,
                    '#hands': len(landmarks.get('hand_landmarks', [])) if landmarks.get('hand_landmarks') else 0,
                    '#pose': len(landmarks.get('pose_landmarks', [])) if landmarks.get('pose_landmarks') else 0
                }
            stats[frame] = presence
            
            # Update max counts
            for key in max_counts:
                max_counts[key] = max(max_counts[key], presence[key])
        
        # Add max counts to stats
        stats['max'] = max_counts
    
    # Save the updated stats
    with open(stats_json_path, 'w') as f:
        json.dump(stats, f)

def load_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def is_string_in_file(file_path, target_string):
    try:
        with Path(file_path).open("r") as f:
            for line in f:
                if target_string in line:
                    return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def resize_frame(frame, frame_size):
    return cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)

def crop_frame(image, bounding_box):
    x, y, w, h = bounding_box
    cropped_frame = image[y:y + h, x:x + w]
    return cropped_frame

def process_landmarks(landmarks, image_shape):
    ih, iw, _ = image_shape
    landmarks_px = np.array([(int(l.x * iw), int(l.y * ih)) for l in landmarks])
    x, y, w, h = cv2.boundingRect(landmarks_px)
    scale_factor = 1.2
    w_padding = int((scale_factor - 1) * w / 2)
    h_padding = int((scale_factor - 1) * h / 2)
    x -= w_padding
    y -= h_padding
    w += 2 * w_padding
    h += 2 * h_padding
    return x, y, w, h

def detect_holistic(image):
    results = mp_holistic.process(image)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    face_prediction = face_detector.detect(mp_image)
    hand_prediction = hand_detector.detect(mp_image)

    bounding_boxes = {}
    landmarks_data = {}

    if face_prediction.face_landmarks:
        bounding_boxes['#face'] = len(face_prediction.face_landmarks) 
        print(f"Number of Faces: {len(face_prediction.face_landmarks)}")

        landmarks_data['face_landmarks'] = []
        for face in face_prediction.face_landmarks:
            landmarks_land = []
            for landmark in face:
                landmarks_land.append([landmark.x, landmark.y, landmark.z])
            landmarks_data['face_landmarks'].append(landmarks_land)        
    else:
        bounding_boxes['#face'] = 0
        landmarks_data['face_landmarks'] = None

    if hand_prediction.hand_landmarks:
        bounding_boxes['#hands'] = len(hand_prediction.hand_landmarks)
        landmarks_data['hand_landmarks'] = []
        for hand in hand_prediction.hand_landmarks:
            landmarks_hand = []
            for landmark in hand:
                landmarks_hand.append([landmark.x, landmark.y, landmark.z])
            landmarks_data['hand_landmarks'].append(landmarks_hand)
    else:
        bounding_boxes['#hands'] = 0
        landmarks_data['hand_landmarks'] = None

    if results.pose_landmarks:
        bounding_boxes['#pose'] = 1
        landmarks_data['pose_landmarks'] = []
        landmarks_data['pose_landmarks'].append([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
    else:
        bounding_boxes['#pose'] = 0
        landmarks_data['pose_landmarks'] = None

    return bounding_boxes, landmarks_data

def video_holistic(video_file, problem_file_path, pose_path, stats_path):
    try:
        video = decord.VideoReader(video_file)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error: {video_file}")
        with (Path(problem_file_path)).open("a") as p:
            p.write(video_file + "\n")        
        return
        
    result_dict = {}
    stats = {}
    fps = video.get_avg_fps()

    landmark_json_path = pose_path / Path(f"{video_file.split('/')[-1].rsplit('.', 1)[0]}_pose.json")
    stats_json_path = stats_path / Path(f"{video_file.split('/')[-1].rsplit('.', 1)[0]}_stats.json")
    
    for i in range(len(video)):        
        try:
            frame_rgb = video[i].asnumpy() # it is actually bgr here
            video.seek(0)
            bounding_boxes, result_dict[i] = detect_holistic(frame_rgb)
        except Exception as e:
            print(f"Error: {e}")
            result_dict[i] = None
            continue
        
        stats[i] = bounding_boxes
    
    with open(landmark_json_path, 'w') as rd:
        json.dump(result_dict, rd)
    with open(stats_json_path, 'w') as st:
        json.dump(stats, st)    
    
         


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--pose_path', type=str, required=True,
                        help='path to where the pose data will be saved')
    parser.add_argument('--stats_path', type=str, required=True,
                        help='path to where the stats data will be saved')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='time limit')
    parser.add_argument('--files_list', type=str, required=True,
                        help='files list')
    parser.add_argument('--problem_file_path', type=str, required=True,
                        help='problem file path')
    parser.add_argument('--face_model_path', type=str, required=True,
                        help='face model path')
    parser.add_argument('--hand_model_path', type=str, required=True,
                        help='hand model path')

    
    args = parser.parse_args()
    index = args.index
    batch_size = args.batch_size
    time_limit = args.time_limit
    files_list = args.files_list
    problem_file_path = args.problem_file_path
    pose_path  = Path(args.pose_path)
    stats_path = Path(args.stats_path)
    face_model_path = args.face_model_path
    hand_model_path = args.hand_model_path

    start_time = time.time()

    global face_detector, hand_detector, mp_holistic

    # Initialize face and hand landmarker models
    base_options_face = python.BaseOptions(model_asset_path=face_model_path)
    options_face = vision.FaceLandmarkerOptions(base_options=base_options_face,
                                                output_face_blendshapes=True,
                                                output_facial_transformation_matrixes=True,
                                                num_faces=6)
    face_detector = vision.FaceLandmarker.create_from_options(options_face)

    base_options_hand = python.BaseOptions(model_asset_path=hand_model_path)
    options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand,
                                                num_hands=6,
                                                min_hand_detection_confidence=0.05)
    hand_detector = vision.HandLandmarker.create_from_options(options_hand)

    # Initialize holistic model
    mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.1)


   




    fixed_list = load_file(files_list)    

    # create folders if they do not exist
    Path(pose_path).mkdir(parents=True, exist_ok=True)
    Path(stats_path).mkdir(parents=True, exist_ok=True)

    # create files if they do not exist
    if not os.path.exists(problem_file_path):   
        with open(problem_file_path, 'w') as f:
            pass
         
    video_batches = [fixed_list[i:i + batch_size] for i in range(0, len(fixed_list), batch_size)]
    for video_file in video_batches[index]:
        current_time = time.time()
        if current_time - start_time > time_limit:
            print("Time limit reached. Stopping execution.")
            break
        
        landmark_json_path = pose_path / Path(f"{video_file.split('/')[-1].rsplit('.', 1)[0]}_pose.json")
        stats_json_path = stats_path / Path(f"{video_file.split('/')[-1].rsplit('.', 1)[0]}_stats.json")
       
        if Path(landmark_json_path).exists() and Path(stats_json_path).exists():
            continue
        elif is_string_in_file(problem_file_path, video_file):
            continue
        else:
            video_holistic(video_file, problem_file_path, pose_path, stats_path)
