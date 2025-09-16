# neon-pupil-lab-eye-tracking-analysis

This repository allows you to preprocess, segment, and analyze videos and eye fixation data from Neon/Pupil Labs, associating each fixation with a segmented object using a DeepLab COCO model.

> **This project is also inspired by [google-research/deeplab2](https://github.com/google-research/deeplab2).**

## Folder Contents

- `scripts/preprocess_neon.py`  
  Preprocesses a Neon/Pupil Labs ZIP export:  
  - Unzips the archive  
  - Extracts frames from the video  
  - Applies undistortion to images and fixations  
  - Generates all files needed for segmentation (undistorted images, corrected fixations, etc.)

- `scripts/segmentation.py`  
  Segments the undistorted images using a DeepLab COCO model, associates each fixation with an object label, and exports a summary CSV.  
  Can also save segmented images and images with fixations.

- `automatisation_script.sh`  
  Automates processing of all ZIP files in `test/zip/`:  
  for each ZIP, runs preprocessing then segmentation, and cleans up the temporary folder.

- `.gitignore`  
  Ignores large model files, results, and temporary files.

- `README.md`  
  This file.

## Pretrained Model Installation

Before running segmentation, download the DeepLab COCO model:

```sh
mkdir -p ../pretrained_model
wget -O ../pretrained_model/convnext_large_kmax_deeplab_coco_train.tar.gz "https://storage.googleapis.com/gresearch/tf-deeplab/saved_model/convnext_large_kmax_deeplab_coco_train.tar.gz"
tar -xzvf ../pretrained_model/convnext_large_kmax_deeplab_coco_train.tar.gz -C ../pretrained_model/
```

## Script Usage

### 1. Preprocessing only

To preprocess a Neon/Pupil Labs ZIP export:

```sh
python scripts/preprocess_neon.py --zip_path /path/to/my_export.zip
```

Intermediate files will be created in `../temp/`.

---

### 2. Segmentation only

To segment the preprocessed images and associate fixations:

```sh
python scripts/segmentation.py --result_path ../results
```

Add the `--is_saved` option to also save segmented images and images with fixations.

---

### 3. Automated processing of multiple ZIP files

Place all your ZIP files in `test/zip/` and run:

```sh
bash automatisation_script.sh
```

Each ZIP will be processed automatically (preprocessing + segmentation), and results will be exported to `../results/`.

---

## Folder Structure

- `scripts/` : main Python scripts
- `zip/` : (create this) put all ZIP files to process here
- `../temp/` : temporary/intermediate files (created automatically)
- `../results/` : final results (CSV, etc.)
- `../pretrained_model/` : downloaded DeepLab model

---

## Notes

- The scripts assume the Neon/Pupil Labs export structure is respected.
- The DeepLab model must be downloaded before first use.
- Scripts are designed to be run from the `test/` folder.
- Large model files are not versioned (see `.gitignore`).
- **This project is inspired by [google-research/deeplab2](https://github.com/google-research/deeplab2).**

---

## Example Full Workflow

1. Place your ZIP files in `test/zip/`
2. Download the pretrained model (see above)
3. Run the automation:
   ```sh
   bash automatisation_script.sh
   ```
4. Results will be in `../results/`

---