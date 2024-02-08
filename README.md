# HierarchicalCRF

## Introduction
This repository contains the supplementary code for the journal article titled "Human Perception-Inspired Grain Segmentation Refinement Using Conditional Random Fields". If you use this code in your research, please cite:

https://arxiv.org/abs/2312.09968
<CITATION INFORMATION WILL BE PROVIDED ONCE AVAILABLE>

This README provides an overview of the post-processing steps involved in refining grain segmentation in fine interconnected grain networks. Below, you will find detailed descriptions of the terminology used, individual scripts, and their functions in the project. In addition, a sample pipeline is provided in the 'sample_vision_processing_pipeline' folder. The details of this pipeline is provided after the detailed descriptions.

## Dependencies
To run the scripts, you will need the following dependencies:
numpy, matplotlib, scipy, scikit-image, tensorflow, networkx, PIL, cv2, joblib, tqdm, [pydensecrf](https://github.com/lucasb-eyer/pydensecrf)

The specific versions can be found in the `requirements.txt`, and can be installed with:

`pip install -r requirements.txt`

Followed by:

`pip install git+https://github.com/lucasb-eyer/pydensecrf.git`

## Installation Instructions
Clone this repository to your local machine. No further installation is required. Run scripts from Post1 to Post5 in sequential order.

## Usage
The post-processing procedure requires two images: the original image (img.tif), and the predicted segmentation mask from a computer vision algorithm (pred.png). It also requires the fragmented segmentation ground truth (FSGT), which are represented as comma-separated values corresponding to the labels associated with broken and completion segments. These segments are identified in Post3_SegmentBasedCRF.py, which are then subsequently manually selected from a list of viable candidates. The .csv files represent the 'on' or 'active' segments. Run the scripts from Post1_PixelBasedCRF.py to Post5_Segmentation.py in order to replicate the results presented in the paper. Each script can also be run individually for debugging purposes. 

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

## Sample Vision Processing Pipeline

In addition to the main post-processing scripts provided in this repository, we offer a sample pipeline located in the `sample_vision_processing_pipeline` folder. This sample demonstrates a complete workflow from image segmentation to obtaining line scan points for subsequent analysis with EELS and EBSD techniques. It also includes a human-in-the-loop (HITL) approach, where after solving the CRF, the user is prompted to add or remove segments.

### Overview

The sample pipeline includes:
- A machine vision model that processes an input image (`img.tif`) to output a segmentation mask (`pred.png`).
- Post-processing scripts (`Post1` through `Post5`) similar to those in the main folder but with modifications specific to the sample pipeline.
- A new script, `Post6_LineScanPoints.py`, exclusive to this pipeline. It labels completion segments, orders points within each segment, and calculates midpoints or points at specified fractions along each completion segment. It then finds a number of points on a line perpendicular to these midpoints, separated by a specified number of pixels. The user can control these variables. The script outputs a `scan_points.csv` file containing the coordinates of these points for further analysis.
- Human-in-the-Loop (HITL) Approach

### Human-in-the-Loop (HITL) Approach
The HITL approach allows manual intervention in the segmentation refinement process. Users can review the `all_viable_paths.png` and `crf_predicted_paths.png` images to identify individual segments. Based on this review, segments can be added or removed using four CSV files:

- `broken_to_add.csv` / `broken_to_remove.csv`
- `completion_to_add.csv` / `completion_to_remove.csv`

These files should contain comma-separated indices of segments, as identified from the `all_viable_paths.png`. This manual step enables users to fine-tune the segmentation results before proceeding with the final steps of the pipeline.

### Key Differences

The scripts within the `sample_vision_processing_pipeline` folder are tailored to provide a straightforward example of how the post-processing pipeline can be applied from start to finish using a provided bash script. While these scripts follow a similar path to those in the main folder, they contain differences to accommodate the sample pipeline's specific needs and the inclusion of the `Post6_LineScanPoints.py` script. For example, processes such as learning weights or comparisons with ground truths are not included.

### Running the Sample Pipeline

This pipeline is designed to be executed from beginning to end using only the provided bash script. It simplifies the process of applying the machine vision model and post-processing steps, culminating in the generation of line scan points.

### Note

It's important to note that the scripts in the `sample_vision_processing_pipeline` folder differ from those in the main folder. Users should refer to this sample for an integrated example of the pipeline's application. The pipeline has its own README file in the folder, providing detailed instructions and explanations specific to the sample example.

We encourage users to explore this sample pipeline to better understand the application of the hierarchical CRF model and its post-processing steps for grain segmentation refinement. 

## License
This project is licensed under the MIT License.

## Contact Information
For any queries or contributions, please contact [daksoy@uci.edu](daksoy@uci.edu).
