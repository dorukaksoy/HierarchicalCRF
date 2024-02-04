# HierarchicalCRF

## Introduction
This repository contains the supplementary code for the journal article titled "Human Perception-Inspired Grain Segmentation Refinement Using Conditional Random Fields". If you use this code in your research, please cite:

https://arxiv.org/abs/2312.09968
<CITATION INFORMATION WILL BE PROVIDED ONCE AVAILABLE>

This README provides an overview of the post-processing steps involved in refining grain segmentation in fine interconnected grain networks. Below, you will find detailed descriptions of the terminology used, individual scripts, and their functions in the project.

## Dependencies
To run the scripts, you will need the following dependencies:
numpy, matplotlib, scipy, skimage, tensorflow, networkx, PIL, cv2, joblib, tqdm, pydensecrf

Install pydensecrf from [here](https://github.com/lucasb-eyer/pydensecrf)

## Installation Instructions
Clone this repository to your local machine. No further installation is required. Run scripts from Post1 to Post5 in sequential order.

## Usage
The post-processing procedure requires two images: the original image (img.tif), and the predicted segmentation mask from a computer vision algorithm (pred.png). It also requires the fragmented segmentation ground truth (FSGT), which are represented as comma-separated values corresponding to the labels associated with broken and completion segments. These segments are identified in Post3_SegmentBasedCRF.py, which are then subsequently manually selected from a list of viable candidates. The .csv files represent the 'on' or 'active' segments. Run the scripts from Post1_PixelBasedCRF.py to Post5_Segmentation.py in order. Each script can also be run individually for debugging purposes.

### Post1_PixelBasedCRF.py
Applies bilateral and Gaussian filters to the image.

### Post2_Classifier.py
Classifies each pixel in the image as either triple junction, isolated point, or broken or complete segments.

### Post3_SegmentBasedCRF.py
Applies segment-based conditional random fields to the image.

### Post4_Pathfinding.py
Traces paths between viable pairs using a cost-based A* pathfinding algorithm. Continually calls TracePaths.py to obtain the traced paths. TracePaths.py generates a 2D Gaussian distribution as the base pattern, then orients it according to source and sink angles to create cost maps for the A* algorithm.

### Post5_Segmentation.py
Applies watershed segmentation to pred_before (predicted mask before post-processing), fsgt (fragmented segmentation ground_truth), and pred_after (predicted mask after post-processing).

### Other scripts and libraries
- TracePaths: See Post4 description.
- CRFLib: Contains methods for the solution of the MILP.
- PostProcessingLib: Includes common methods used across scripts.
- PostProcessingPlot: Methods for creating plots.
- AstarLib: Necessary methods for the cost-based A* algorithm.

## Terminology
- *Tag*: Marking a specific feature in the image. Pixels are marked 1 (True) for that feature, and 0 (False) otherwise. E.g., tagged_triple_junctions.
- *Label*: Used when differentiating between specific segments is important, such as identifying different broken segments. These are unique identification numbers shared by each pixel on a specific segment. E.g., labeled_broken_segments (an image where each pixel is tagged according to the segment they are on) and broken_segment_labels (a list including the labels for each broken segment in the image).
- *Classify*: Marking each pixel in an image with feature-specific identifiers, i.e., {Complete segments : 1, Triple junctions : 2, Broken segments : 3, Isolated Point : 4}.

We welcome contributions and suggestions to improve this project. Please feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License.

## Contact Information
For any queries or contributions, please contact [daksoy@uci.edu](daksoy@uci.edu).
