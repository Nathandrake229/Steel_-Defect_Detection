# Steel_-Defect_Detection
Defect detection and masking model for Steel Microstructure

Used on Severstal Steel Dataset on Kaggle

Mask RCNN Model
  Create a virtual environment and install packages from requirements.txt

  Use mask_rcnn_coco.h5 file for pretrained weights

  Train command - python met_defect_Mask_RCNN.py train --dataset=train_1.csv --subset=train --weights=coco

  Detect command - python met_defect_Mask_RCNN.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5
  
  
UNet model
  
  Run command - python met_defect_UNET.py
