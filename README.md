# Neon pupil Lab Eye-tracking analysis

This repository provides tools to preprocess, segment, and analyze videos and eye fixation data recorded with [Pupil Labs Neon](https://pupil-labs.com/products/neon) eye-tracking glasses. The main goal is to associate each eye fixation with a segmented object in the scene, using a DeepLab COCO model, and to export a CSV file linking each video frame, fixation, and object label.

> **This project is inspired by [google-research/deeplab2](https://github.com/google-research/deeplab2).**

---

## Project Purpose

The main objective of this codebase is to facilitate the analysis of real-world gaze behavior by:
- **Preprocessing** raw data exported from Pupil Labs Neon (video + timeseries data).
- **Segmenting** each video frame using a state-of-the-art semantic segmentation model (DeepLab).
- **Associating** each eye fixation with the corresponding object in the scene.
- **Exporting** a CSV file that, for each frame, links the fixation and the detected object label.

This enables researchers to study what objects participants are looking at, frame by frame, in naturalistic environments.

---

## About the Data & Hardware

- **Input:** ZIP files **directly downloaded from Pupil Cloud** using the "Timeseries Data + Scene video" export option.
- **Hardware:** [Pupil Labs Neon](https://pupil-labs.com/products/neon) glasses, which provide high-precision eye-tracking synchronized with egocentric video.
- **Output:** A CSV file for each recording, associating each frame with a fixation and the corresponding object label (as detected by the segmentation model).

---

## Model Used

We use the **DeepLab ConvNeXt Large KMaX** model, trained on the COCO dataset, for semantic segmentation. 

---

## Repository Contents

- `scripts/preprocess_neon.py`  
  Preprocesses a Neon/Pupil Labs ZIP export:
  - Unzips the archive
  - Extracts frames from the video
  - Applies undistortion to images and fixations
  - Generates all files needed for segmentation

- `scripts/segmentation.py`  
  Segments the undistorted images using the DeepLab COCO model, associates each fixation with an object label, and exports a summary CSV.  
  Optionally saves segmented images and images with fixations.

- `automatisation_script.sh`  
  Automates processing of all ZIP files in a specified folder:  
  For each ZIP, runs preprocessing then segmentation, and cleans up the temporary folder.

---

## Pretrained Model Installation

Before running segmentation, download the DeepLab COCO model:

```sh
mkdir -p pretrained_model &&
wget -O pretrained_model/convnext_large_kmax_deeplab_coco_train.tar.gz \
https://storage.googleapis.com/gresearch/tf-deeplab/saved_model/convnext_large_kmax_deeplab_coco_train.tar.gz &&
tar -xzf pretrained_model/convnext_large_kmax_deeplab_coco_train.tar.gz -C pretrained_model
```

---

## Script Usage

### 1. Preprocessing only

To preprocess a ZIP export from Pupil Cloud:

```sh
python scripts/preprocess_neon.py --zip_path /path/to/my_export.zip
```

Intermediate files will be created in `../temp/`.

---

### 2. Segmentation only

To segment the preprocessed images and associate fixations:

```sh
python scripts/segmentation.py --result_path results
```

Add the `--is_saved` option to also save segmented images and images with fixations.

---

### 3. Automated processing of multiple ZIP files

**Before running the automation script, edit the `ZIP_DIR` variable in `automatisation_script.sh` to point to the folder containing all your ZIP files downloaded from Pupil Cloud ("Timeseries Data + Scene video" option).**

By default, results will be saved in the `results/` folder.  
If you want to change the output folder, modify the `RESULT_DIR` variable in the same script.

Then, from the project directory, run:

```sh
bash automatisation_script.sh
```

Each ZIP will be processed automatically (preprocessing + segmentation), and the results (CSV associating frame, fixation, and object label) will be exported to the chosen folder.

---

## Folder Structure

- `scripts/` : main Python scripts
- `zip/` : (create this) put all ZIP files to process here
- `temp/` : temporary/intermediate files (created automatically)
- `results/` : final results (CSV, etc.)
- `pretrained_model/` : downloaded DeepLab model

---

## Notes

- The scripts assume the Neon/Pupil Labs export structure is respected.
- The DeepLab model must be downloaded before first use.
- **This project is inspired by [google-research/deeplab2](https://github.com/google-research/deeplab2).**

---

## Example Full Workflow

1. Download your ZIP files from Pupil Cloud ("Timeseries Data + Scene video" option)
2. Place them in your chosen folder (update `ZIP_DIR` in the script)
3. Download the pretrained model (see above)
4. Run the automation:
   ```sh
   bash automatisation_script.sh
   ```
5. Results (CSV associating frame, fixation, and object label) will be in `../results/` (or your chosen output folder)

---
