import os
import zipfile
import glob
import subprocess
import json
import cv2
import numpy as np
import csv
import argparse
from tqdm import tqdm

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

def extract_frames(video_path, output_folder, fps=30):
    os.makedirs(output_folder, exist_ok=True)
    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,
        "-r", str(fps),
        "-vframes", "400",
        f"{output_folder}/img%04d.jpg"
    ]
    subprocess.run(ffmpeg_command, check=True)

def undistort_frames(frames_folder, scene_camera_json, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    with open(scene_camera_json, 'r') as f:
        scene_camera = json.load(f)
    K = np.array(scene_camera['camera_matrix'])
    dist_coeffs = np.array(scene_camera['distortion_coefficients'])
    
    img_files = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))

    for img_file in tqdm(img_files, desc="Undistorting frames", unit="image"):
        img = cv2.imread(img_file)
        undistorted = cv2.undistort(img, K, dist_coeffs, None, K)
        out_path = os.path.join(output_folder, os.path.basename(img_file))
        cv2.imwrite(out_path, undistorted)
        
def undistort_fixations(fixations, K, dist_coeffs):
    corrected_fixations = []
    for fixation in tqdm(fixations, desc="Undistorting fixations", unit="fix"):
        point = np.array([[fixation['x'], fixation['y']]], dtype=np.float32)
        undistorted_point = cv2.undistortPoints(point, K, dist_coeffs, None, K)
        corrected_fixations.append({
            "start": fixation['start'],
            "end": fixation['end'],
            "duration": fixation['duration'],
            "x": undistorted_point[0][0][0],
            "y": undistorted_point[0][0][1]
        })
    return corrected_fixations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, required=True)
    args = parser.parse_args()
    
    # Create a temporary folder for results
    temporary_folder = "../temp/"
    os.makedirs(temporary_folder, exist_ok=True)
    
    data_folder = os.path.join(temporary_folder, "data")

    extract_zip(args.zip_path, data_folder)
    # Find the path to the video and the json
    print("Searching for extracted files...")
    extracted_dirs = [name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))]
    unique_folder_path = os.path.join(data_folder, extracted_dirs[0])
    video_path = glob.glob(os.path.join(data_folder, "**/*.mp4"), recursive=True)[0]
    scene_camera_json = os.path.join(unique_folder_path, "scene_camera.json")
    
    frame_folder = os.path.join(temporary_folder, "frames")
    undistorted_frame_folder = os.path.join(temporary_folder, "frames_undistorted")

    extract_frames(video_path, frame_folder, fps=30)
    undistort_frames(frame_folder, scene_camera_json, undistorted_frame_folder)

    # Undistortion of fixations
    fixations_path = os.path.join(unique_folder_path, "fixations.csv")
    fixations = []
    with open(fixations_path, "r") as csvfile:
        reader = csv.reader(csvfile)

        # Skip first line
        next(reader)
        
        #reader = csv.DictReader(csvfile)
        for row in reader:
            fixations.append({
                "start": float(row[3]),
                "end": float(row[4]),
                "duration": float(row[5]),
                "x": float(row[6]),
                "y": float(row[7])
            })

    # Load camera parameters
    with open(scene_camera_json, 'r') as f:
        scene_camera = json.load(f)
    K = np.array(scene_camera['camera_matrix'])
    dist_coeffs = np.array(scene_camera['distortion_coefficients'])

    fixations_undistorted = undistort_fixations(fixations, K, dist_coeffs)

    # Save to a new CSV
    undistorted_path = os.path.join(temporary_folder, "fixations_undistorted.csv")
    with open(undistorted_path, "w", newline="") as csvfile:
        fieldnames = ["start", "end", "duration", "x", "y"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for f in fixations_undistorted:
            writer.writerow(f)
            
    # Read info.json to get start_time and save it
    info_json_path = os.path.join(unique_folder_path, "info.json")
    with open(info_json_path, 'r') as f:
        info = json.load(f)
    start_time = info['start_time']
    for _ in tqdm(range(1), desc="Writing start_time"):
        with open(os.path.join(temporary_folder, "start_time.txt"), "w") as f:
            f.write(str(start_time))
    
    # Get the name of the unique folder in "data"
    extracted_dirs = [name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))]
    name_of_file = extracted_dirs[0]
    for _ in tqdm(range(1), desc="Writing name_of_file"):
        with open(os.path.join(temporary_folder, "name_of_file.txt"), "w") as f:
            f.write(str(name_of_file))