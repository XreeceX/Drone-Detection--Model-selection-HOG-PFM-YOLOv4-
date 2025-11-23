# Drone Detection System - MATLAB Implementation

This project implements three different algorithms for drone detection using MATLAB: YOLOv4, Histograms of Oriented Gradients (HOG), and Point Feature Matching. The code is organized in a modular structure with separate files for each algorithm and support for multiple datasets.

## Project Structure

### Main Program
- `main_drone_detection.m` - Main program file with configuration section at the top

### Algorithm Files (Separate files for each algorithm)
- `yolov4_detector.m` - YOLOv4 deep learning algorithm
- `hog_detector.m` - Histograms of Oriented Gradients (HOG) with SVM classifier
- `point_feature_matching_detector.m` - Point Feature Matching (SIFT/SURF/ORB)

### Dataset Files
- `load_dataset_images.m` - Loads images from a dataset folder

## Features

1. **Three Detection Algorithms**:
   - **YOLOv4** - Deep learning-based object detection
   - **HOG** - Histograms of Oriented Gradients with SVM classifier
   - **Point Feature Matching** - Template matching using SIFT/SURF/ORB features

2. **Three Dataset Support**:
   - `dataset1` - First dataset folder
   - `dataset2` - Second dataset folder
   - `dataset3` - Third dataset folder

3. **Configurable Parameters** (set in main program):
   - Algorithm selection
   - Dataset selection
   - Detection thresholds
   - Display and save options

## Requirements

- MATLAB R2020b or later
- Deep Learning Toolbox (for YOLOv4)
- Computer Vision Toolbox (for HOG and Point Feature Matching)
- Statistics and Machine Learning Toolbox (for HOG SVM classifier)

## Usage

1. **Configure the main program** (`main_drone_detection.m`):
   - Open `main_drone_detection.m`
   - Set the configuration parameters at the top:
     ```matlab
     selectedAlgorithm = 'yolov4';  % or 'hog' or 'point_feature_matching'
     selectedDataset = 'dataset1';  % or 'dataset2' or 'dataset3'
     
     % ADD YOUR DATASET PATHS HERE
     dataset1Path = '';  % Add path to dataset 1 folder
     dataset2Path = '';  % Add path to dataset 2 folder
     dataset3Path = '';  % Add path to dataset 3 folder
     
     confThreshold = 0.5;  % Confidence threshold (for YOLOv4)
     iouThreshold = 0.4;   % IoU threshold for NMS (for YOLOv4)
     hogThreshold = 0.3;    % HOG detection threshold
     pointFeatureThreshold = 0.6;  % Point feature matching threshold
     ```

2. **Run the main program**:
   ```matlab
   main_drone_detection
   ```

## Dataset Preparation

### Image Datasets:
- Create three folders containing your drone images
- Supported formats: JPG, JPEG, PNG, BMP, TIF, TIFF
- Set `dataset1Path`, `dataset2Path`, and `dataset3Path` in the main program
- Each dataset folder should contain images to be processed

## Model Files

### YOLOv4:
- `yolov4ModelPath` - Path to YOLOv4 model file (e.g., `'yolov4_drone_model.mat'`)
- The model should contain a trained `yolov4ObjectDetector` object

### HOG:
- `hogModelPath` - Path to HOG-SVM model file (e.g., `'hog_drone_model.mat'`)
- The model should contain:
  - `classifier` - Trained SVM classifier
  - `hogParams` - HOG parameters (optional)

### Point Feature Matching:
- `pointFeatureTemplatePath` - Path to template image or feature file
- Can be:
  - Image file (JPG, PNG, etc.) - features will be extracted automatically
  - MAT file containing pre-computed features (`templateImg`, `features`, `points`)

**Note**: The current implementation includes placeholder detection functions. To use actual detection:

1. **For YOLOv4**: Train a YOLOv4 model on your drone dataset or use a pre-trained model
2. **For HOG**: Train an SVM classifier using HOG features extracted from training images
3. **For Point Feature Matching**: Provide a template image of a drone

## Output

- Detection results are displayed on images (if `displayResults = true`)
- Detection statistics are printed to the console:
  - Total drones detected
  - Total processing time
  - Number of images processed
  - Average time per image
- Results can be saved to a .mat file (if `saveResults = true`)

## Algorithm Details

### YOLOv4
- Deep learning-based object detection
- Uses pre-trained or custom-trained YOLOv4 model
- Applies Non-Maximum Suppression (NMS) to remove overlapping detections
- Output: Bounding boxes with confidence scores

### HOG (Histograms of Oriented Gradients)
- Feature-based detection using HOG descriptors
- Uses SVM classifier for drone/non-drone classification
- Sliding window approach for detection
- Output: Bounding boxes with confidence scores

### Point Feature Matching
- Template-based detection using point features (SIFT/SURF/ORB)
- Matches features between template and input image
- Uses geometric transformation for robust matching
- Output: Bounding boxes with match confidence

## File Organization

Each algorithm is in its own separate file, making it easy to:
- Add new algorithms by creating new files following the same structure
- Modify individual algorithms without affecting others
- Maintain and debug code more easily
- Compare performance between different algorithms

## Customization

To use actual detection instead of placeholders:

1. **YOLOv4**: Train or load a YOLOv4 model and update `load_yolov4_network()` function
2. **HOG**: Train an SVM classifier with HOG features and save it to the model path
3. **Point Feature Matching**: Provide a template image of a drone at the template path

## Notes

- The code includes comprehensive error handling
- Processing time is tracked and reported for each algorithm
- Detection results include bounding boxes and confidence scores
- All paths and parameters are configured in the main program file for easy access
- Dataset paths are left blank in the configuration - add your paths before running

## License

This code is provided as-is for educational and research purposes.
