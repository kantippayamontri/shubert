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

    if x + w > iw:
        x = iw - w
    
    if y + h > ih:
        y = ih - h
    
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
    nose_landmark_from_pose = pose_landmarks[0]
    nose_landmarks_from_face = []
    for i in range(0, len(face_landmarks)):
        nose_landmarks_from_face.append(face_landmarks[i][0])
    closest_nose_index = np.argmin([np.linalg.norm(np.array(nose_landmark_from_pose) - np.array(nose_landmark)) for nose_landmark in nose_landmarks_from_face])
    return face_landmarks[closest_nose_index]

def select_hands(pose_landmarks, hand_landmarks, image_shape):
    
    if hand_landmarks is None:
        return None, None
    left_wrist_from_pose = pose_landmarks[15]
    right_wrist_from_pose = pose_landmarks[16]
    ih, iw, _ = image_shape          
    wrist_from_hand = []    
    for i in range(0, len(hand_landmarks)):
        wrist_from_hand.append(hand_landmarks[i][0])
    
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

def calculate_bounding_box(landmarks, indices, image_shape):
    x_coordinates = [landmarks[i][0] for i in indices]
    y_coordinates = [landmarks[i][1] for i in indices]

    left = min(x_coordinates)
    right = max(x_coordinates)
    top = min(y_coordinates)
    bottom = max(y_coordinates)

    return int(left * image_shape[1]), int(top * image_shape[0]), \
           int(right * image_shape[1]), int(bottom * image_shape[0])


def cues_on_grey_background(image, facial_landmarks):
    image_shape = image.shape
    left_eye_indices = [69, 168, 156, 118, 54]
    right_eye_indices = [168, 299, 347, 336, 301]
    mouth_indices = [164, 212, 432, 18]
    
    # Calculate bounding boxes
    left_eye_box = calculate_bounding_box(facial_landmarks, left_eye_indices, image_shape)
    right_eye_box = calculate_bounding_box(facial_landmarks, right_eye_indices, image_shape)
    mouth_box = calculate_bounding_box(facial_landmarks, mouth_indices, image_shape)

    # Calculate the overall bounding box
    min_x = min(left_eye_box[0], right_eye_box[0], mouth_box[0])
    min_y = min(left_eye_box[1], right_eye_box[1], mouth_box[1])
    max_x = max(left_eye_box[2], right_eye_box[2], mouth_box[2])
    max_y = max(left_eye_box[3], right_eye_box[3], mouth_box[3])

    # Add some padding
    padding = 10
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(image.shape[1], max_x + padding)
    max_y = min(image.shape[0], max_y + padding)

    # Make the crop a square by adjusting either width or height
    width = max_x - min_x
    height = max_y - min_y
    side_length = max(width, height)

    # Adjust min_x, min_y, max_x, max_y to ensure square
    if width < side_length:
        # Expand width equally on both sides
        extra = side_length - width
        min_x = max(0, min_x - extra // 2)
        max_x = min(image.shape[1], max_x + extra // 2)

    if height < side_length:
        # Expand height equally on both sides
        extra = side_length - height
        min_y = max(0, min_y - extra // 2)
        max_y = min(image.shape[0], max_y + extra // 2)

    # Create a grey background image of the square size
    grey_background = np.ones((side_length, side_length, 3), dtype=np.uint8) * 128

    # Function to crop and paste region
    def crop_and_paste(src, dst, src_box, dst_origin):
        x1, y1, x2, y2 = src_box
        dx, dy = dst_origin
        crop = src[y1:y2, x1:x2]
        crop_height, crop_width = crop.shape[:2]
        dst[dy:dy+crop_height, dx:dx+crop_width] = crop

    # Crop and paste regions onto grey background
    crop_and_paste(image, grey_background, left_eye_box, (left_eye_box[0]-min_x, left_eye_box[1]-min_y))
    crop_and_paste(image, grey_background, right_eye_box, (right_eye_box[0]-min_x, right_eye_box[1]-min_y))
    crop_and_paste(image, grey_background, mouth_box, (mouth_box[0]-min_x, mouth_box[1]-min_y))

    return grey_background
   

def video_holistic(video_file, face_path, problem_file_path, pose_path):
    video = decord.VideoReader(video_file)

    fps = video.get_avg_fps()

    clip_face_path = Path(face_path) / f"{video_file.split('/')[-1].rsplit('.', 1)[0]}_face.mp4"
    landmark_json_path = Path(pose_path) / Path(f"{video_file.split('/')[-1].rsplit('.', 1)[0]}_pose.json")

    #Make parent directories for clip_face_path if they do not exist
    clip_face_path.parent.mkdir(parents=True, exist_ok=True)
    # remove the files if they already exist
    if os.path.exists(clip_face_path):
        os.remove(clip_face_path)

    fourcc_face = cv2.VideoWriter_fourcc(*'mp4v')
    out_face = cv2.VideoWriter(clip_face_path, fourcc_face, fps, (224, 224))

    with open(landmark_json_path, 'r') as rd:
        result_dict = json.load(rd)

    prev_face_frame = None
    prev_result_dict = None

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
            if prev_face_frame is not None:
                out_face.write(prev_face_frame)
            else:
                face_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                out_face.write(face_frame)                
            continue
        
                       
        # it contains a body pose that can be use as reference to get the face 
        if result_dict[str(i)]['face_landmarks'] is not None: # some face_landmarks were detected
            face_lmks = select_face(result_dict[str(i)]['pose_landmarks'][0], result_dict[str(i)]['face_landmarks']) # from all the faces, select the one that is closer to the face of the pose landmark
            face_frame = cues_on_grey_background(frame_rgb, face_lmks)            
            face_frame = resize_frame(face_frame, (224, 224))            
            out_face.write(face_frame)
            prev_face_frame = face_frame

        elif prev_face_frame is not None:
            out_face.write(prev_face_frame) 
        else:
            # use a blank frame if no face_landmarks or body_landmarks were detected
            face_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            out_face.write(face_frame)

    out_face.release()
    del out_face
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
    parser.add_argument('--face_path', type=str, required=True,
                        help='face path')

    start_time = time.time()

    
    args = parser.parse_args()
    index = args.index
    batch_size = args.batch_size
    time_limit = args.time_limit
    files_list = args.files_list
    problem_file_path = args.problem_file_path
    pose_path = args.pose_path
    face_path = args.face_path
    
    fixed_list = load_file(files_list)

    if not os.path.exists(problem_file_path):
        with (Path(problem_file_path)).open("w") as f:
            f.write("")
            
    video_batches = [fixed_list[i:i + batch_size] for i in range(0, len(fixed_list), batch_size)]
    for video_file in video_batches[index]:
        current_time = time.time()
        if current_time - start_time > time_limit:
            print("Time limit reached. Stopping execution.")
            break
        
        clip_face_path = Path(face_path) / f"{video_file.split('/')[-1].rsplit('.', 1)[0]}_face.mp4"
        
        if os.path.exists(clip_face_path):
            print(f"File {clip_face_path} already exists. Skipping.")
            continue
        elif is_string_in_file(problem_file_path, video_file):
            print(f"File {video_file} is in the problem file. Skipping.")
            continue
        else:
            try:
                video_holistic(video_file, face_path, problem_file_path, pose_path)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error: {video_file}")
                with (Path(problem_file_path)).open("a") as p:
                    p.write(video_file + "\n")
            finally:
                gc.collect()
