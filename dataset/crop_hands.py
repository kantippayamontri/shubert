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
import time
import gc

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
    if frame is not None and frame.size > 0:
        return cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
    else:
        return None

def crop_frame(image, bounding_box):
    x, y, w, h = bounding_box
    cropped_frame = image[y:y + h, x:x + w]
    return cropped_frame

def get_bounding_box(landmarks, image_shape, scale_factor=1.2):
    ih, iw, _ = image_shape
    landmarks_px = np.array([(int(l[0] * iw), int(l[1] * ih)) for l in landmarks])
    center_x, center_y = np.mean(landmarks_px, axis=0, dtype=int)
    xb, yb, wb, hb = cv2.boundingRect(landmarks_px)
    box_size = max(wb, hb)
    half_size = box_size // 2
    x = center_x - half_size
    y = center_y - half_size
    w = box_size
    h = box_size
    
    
    w_padding = int((scale_factor - 1) * w / 2)
    h_padding = int((scale_factor - 1) * h / 2)
    x -= w_padding
    y -= h_padding
    w += 2 * w_padding
    h += 2 * h_padding    
    
    return x, y, w, h

def adjust_bounding_box(bounding_box, image_shape):
    x, y, w, h = bounding_box
    ih, iw, _ = image_shape

    # Adjust x-coordinate if the bounding box extends beyond the image's right edge
    if x + w > iw:
        x = iw - w
    
    # Adjust y-coordinate if the bounding box extends beyond the image's bottom edge
    if y + h > ih:
        y = ih - h
    
    # Ensure bounding box's x and y coordinates are not negative
    x = max(x, 0)
    y = max(y, 0)    

    return x, y, w, h

def get_centered_box(landmarks, image_shape, box_size, scale_factor=1.5):
    ih, iw, _ = image_shape
    landmarks_px = np.array([(int(l[0] * iw), int(l[1] * ih)) for l in landmarks])
    center_x, center_y = np.mean(landmarks_px, axis=0, dtype=int)
    half_size = box_size // 2
    x = center_x - half_size
    y = center_y - half_size
    w = box_size
    h = box_size

    return x, y, w, h


def is_center_inside_frame(landmarks, image_shape):
    ih, iw, _ = image_shape
    landmarks_px = np.array([(int(l[0] * iw), int(l[1] * ih)) for l in landmarks])
    center_x, center_y = np.mean(landmarks_px, axis=0, dtype=int)
    return 0 <= center_x <= iw and 0 <= center_y <= ih


def isl_wrist_below_elbow(pose_landmarks):
    left_elbow = pose_landmarks[13]
    left_wrist = pose_landmarks[15]
    return left_wrist[1] > left_elbow[1]

def isr_wrist_below_elbow(pose_landmarks):
    right_elbow = pose_landmarks[14]
    right_wrist = pose_landmarks[16]
    return right_wrist[1] > right_elbow[1]    

def get_hand_direction(elbow_landmark, wrist_landmark):
    direction = np.array([wrist_landmark[0], wrist_landmark[1]]) - np.array([elbow_landmark[0], elbow_landmark[1]])
    direction = direction / np.linalg.norm(direction)
    return direction

def shift_bounding_box(bounding_box, direction, magnitude):
    x, y, w, h = bounding_box
    shift_x = int(direction[0] * magnitude)
    shift_y = int(direction[1] * magnitude)
    shifted_box = (x + shift_x, y + shift_y, w, h)
    return shifted_box

def select_face(pose_landmarks, face_landmarks):
    # select the nose landmark from the pose landmark.
    nose_landmark_from_pose = pose_landmarks[0]
    # list of nose landamrk(s) obtained from the face landmarks
    nose_landmarks_from_face = []
    for i in range(0, len(face_landmarks)):
        nose_landmarks_from_face.append(face_landmarks[i][0])
    # return the indices of the closest nose from nose_landmarks_from_face to nose_landmark_from_pose
    closest_nose_index = np.argmin([np.linalg.norm(np.array(nose_landmark_from_pose) - np.array(nose_landmark)) for nose_landmark in nose_landmarks_from_face])
    return face_landmarks[closest_nose_index]

def select_hands(pose_landmarks, hand_landmarks, image_shape):
    
    if hand_landmarks is None:
        return None, None
    # select the wrist landmarks from the pose landmark.
    left_wrist_from_pose = pose_landmarks[15]
    right_wrist_from_pose = pose_landmarks[16]

    ih, iw, _ = image_shape        
        
    
    wrist_from_hand = []    
    for i in range(0, len(hand_landmarks)):
        # array of wrist landmarks from the hand landmarks
        wrist_from_hand.append(hand_landmarks[i][0])
    
    # the euclidean distance between the two points using only the first 2 coordinates.
    if right_wrist_from_pose is not None:
        right_hand_landmarks = hand_landmarks[0]
        minimum_distance = 100
        for i in range(0, len(hand_landmarks)):
            distance = np.linalg.norm(np.array(right_wrist_from_pose[0:2]) - np.array(wrist_from_hand[i][0:2]))
            if distance < minimum_distance:
                minimum_distance = distance
                right_hand_landmarks = hand_landmarks[i]
        
        if minimum_distance >= 0.1:
            right_hand_landmarks = None
            
    else:
        right_hand_landmarks = None
    
    if left_wrist_from_pose is not None:
        left_hand_landmarks = hand_landmarks[0]
        minimum_distance = 100
        for i in range(0, len(hand_landmarks)):
            distance = np.linalg.norm(np.array(left_wrist_from_pose[0:2]) - np.array(wrist_from_hand[i][0:2]))
            if distance < minimum_distance:
                minimum_distance = distance
                left_hand_landmarks = hand_landmarks[i]
        if minimum_distance >= 0.1:
            left_hand_landmarks = None
        

    else:
        left_hand_landmarks = None
        
    return left_hand_landmarks, right_hand_landmarks
        


def video_holistic(video_file, hand_path, problem_file_path, pose_path):
    
    video = decord.VideoReader(video_file)
    fps = video.get_avg_fps()

    clip_hand1_path = Path(hand_path) / Path(f"{video_file.split('/')[-1].rsplit('.', 1)[0]}_hand1.mp4")
    clip_hand2_path = Path(hand_path) / Path(f"{video_file.split('/')[-1].rsplit('.', 1)[0]}_hand2.mp4")
    landmark_json_path = Path(pose_path) / Path(f"{video_file.split('/')[-1].rsplit('.', 1)[0]}_pose.json")
    

    if os.path.exists(clip_hand1_path):
        os.remove(clip_hand1_path)
    if os.path.exists(clip_hand2_path):
        os.remove(clip_hand2_path)
    


    fourcc_hand1 = cv2.VideoWriter_fourcc(*'mp4v')
    out_hand1 = cv2.VideoWriter(clip_hand1_path, fourcc_hand1, fps, (224, 224))

    fourcc_hand2 = cv2.VideoWriter_fourcc(*'mp4v')
    out_hand2 = cv2.VideoWriter(clip_hand2_path, fourcc_hand2, fps, (224, 224))

    with open(landmark_json_path, 'r') as rd:
        result_dict = json.load(rd)

    # prev_face_frame = None
    prev_hand1_frame = None
    prev_hand2_frame = None
    prev_result_dict = None

    # face_box_size = None
    hand1_box_size = None
    hand2_box_size = None
    


    for i in range(len(video)):
        frame = video[i].asnumpy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if result_dict[str(i)] is None: # no pose_landmarks detected
            if prev_result_dict is not None: # use the previous pose_landmarks
                result_dict[str(i)] = prev_result_dict
            else: # use a blank frame if no pose_landmarks was ever detected for this clip
                continue
        else: # store pose_landmarks detected as last known pose_landmarks
            prev_result_dict = result_dict[str(i)]

        if result_dict[str(i)]['pose_landmarks'] is None: # some pose_landmarks were detected

            if prev_hand1_frame is not None:
                out_hand1.write(prev_hand1_frame)
            else:
                hand1_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                out_hand1.write(hand1_frame)
            if prev_hand2_frame is not None:
                out_hand2.write(prev_hand2_frame)   
            else:
                hand2_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                out_hand2.write(hand2_frame)
                
            continue
        

        
        
        
        # select the left and right hand from the hand_landmarks
        result_dict[str(i)]['left_hand_landmarks'], result_dict[str(i)]['right_hand_landmarks'] = select_hands(result_dict[str(i)]['pose_landmarks'][0], result_dict[str(i)]['hand_landmarks'], frame_rgb.shape)


        if result_dict[str(i)]['left_hand_landmarks'] is not None:
            hand1_box = get_bounding_box(result_dict[str(i)]['left_hand_landmarks'], frame_rgb.shape, scale_factor=1.5)
            hand1_box = adjust_bounding_box(hand1_box, frame_rgb.shape)
            hand1_frame = crop_frame(frame_rgb, hand1_box)
            hand1_frame = resize_frame(hand1_frame, (224, 224))
            out_hand1.write(hand1_frame)
            prev_hand1_frame = hand1_frame                       
        elif prev_hand1_frame is not None:
            out_hand1.write(prev_hand1_frame)
        else:
            hand1_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            out_hand1.write(hand1_frame)

        if result_dict[str(i)]['right_hand_landmarks'] is not None:
            hand2_box = get_bounding_box(result_dict[str(i)]['right_hand_landmarks'], frame_rgb.shape, scale_factor=1.5)
            hand2_box = adjust_bounding_box(hand2_box, frame_rgb.shape)
            hand2_frame = crop_frame(frame_rgb, hand2_box)
            hand2_frame = resize_frame(hand2_frame, (224, 224))
            out_hand2.write(hand2_frame)
            prev_hand2_frame = hand2_frame                       
        elif prev_hand2_frame is not None:
            out_hand2.write(prev_hand2_frame)
        else:
            hand2_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            out_hand2.write(hand2_frame)



    out_hand1.release()
    out_hand2.release()
    del out_hand1
    del out_hand2
    del video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='time limit')
    parser.add_argument('--files_list', type=str, required=True,
                        help='files list')
    parser.add_argument('--problem_file_path', type=str, required=True,
                        help='problem file path')
    parser.add_argument('--pose_path', type=str, required=True,
                        help='pose path')
    parser.add_argument('--hand_path', type=str, required=True,
                        help='hand path')


    start_time = time.time()

    
    args = parser.parse_args()
    index = args.index
    batch_size = args.batch_size
    time_limit = args.time_limit
    files_list = args.files_list
    problem_file_path = args.problem_file_path
    pose_path = args.pose_path
    hand_path = args.hand_path

    # Create directories if they do not exist
    Path(hand_path).mkdir(parents=True, exist_ok=True)

    
    fixed_list = load_file(files_list)


    
    # create the files if they do not exist
    if not os.path.exists(problem_file_path):
        with (Path(problem_file_path)).open("w") as f:
            f.write("")
            
    video_batches = [fixed_list[i:i + batch_size] for i in range(0, len(fixed_list), batch_size)]
    for video_file in video_batches[index]:      
        current_time = time.time()
        if current_time - start_time > time_limit:
            print("Time limit reached. Stopping execution.")
            break
  
        clip_hand2_path = Path(hand_path) / Path(f"{video_file.split('/')[-1].rsplit('.', 1)[0]}_hand2.mp4")
        
        if Path(clip_hand2_path).exists():        
            continue
        elif is_string_in_file(problem_file_path, video_file):
            continue
        else:
            try:
                video_holistic(video_file, hand_path, problem_file_path, pose_path)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error: {video_file}")
                with (Path(problem_file_path)).open("a") as p:
                    p.write(video_file + "\n")
            finally:
                gc.collect()
            