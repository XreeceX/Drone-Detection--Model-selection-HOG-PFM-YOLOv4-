%% Main Program for Drone Detection
% This program allows selection of algorithm and dataset for drone detection
% Author: Generated for Anika Project
% Date: 2024

clear; clc; close all;

%% ========================================
%% CONFIGURATION SECTION - SET PATHS HERE
%% ========================================

% Algorithm Selection: 'yolov4', 'hog', or 'point_feature_matching'
selectedAlgorithm = 'yolov4';

% Dataset Selection: 'dataset1', 'dataset2', or 'dataset3'
selectedDataset = 'dataset1';

% Dataset Paths (ADD YOUR DATASET PATHS HERE)
% Leave blank and add your dataset folder paths below
dataset1Path = '';  % Add path to dataset 1 folder
dataset2Path = '';  % Add path to dataset 2 folder
dataset3Path = '';  % Add path to dataset 3 folder

% Model Paths (for algorithms)
yolov4ModelPath = 'yolov4_drone_model.mat';  % YOLOv4 model path
hogModelPath = 'hog_drone_model.mat';         % HOG model path (SVM classifier)
pointFeatureTemplatePath = 'point_feature_template.mat';  % Template for point feature matching

% Detection Parameters
confThreshold = 0.5;  % Confidence threshold (0-1) - for YOLOv4
iouThreshold = 0.4;   % IoU threshold for NMS (0-1) - for YOLOv4
hogThreshold = 0.3;   % HOG detection threshold
pointFeatureThreshold = 0.6;  % Point feature matching threshold

% Output Settings
displayResults = true;  % Set to false to skip displaying results
saveResults = false;    % Set to true to save detection results
outputResultsPath = 'detection_results.mat';  % Path to save results

%% ========================================
%% END OF CONFIGURATION SECTION
%% ========================================

%% Display Configuration
fprintf('========================================\n');
fprintf('   Drone Detection System\n');
fprintf('========================================\n\n');
fprintf('Configuration:\n');
fprintf('  Algorithm: %s\n', selectedAlgorithm);
fprintf('  Dataset: %s\n', selectedDataset);

% Get dataset path based on selection
switch lower(selectedDataset)
    case 'dataset1'
        datasetPath = dataset1Path;
    case 'dataset2'
        datasetPath = dataset2Path;
    case 'dataset3'
        datasetPath = dataset3Path;
    otherwise
        error('Unknown dataset: %s. Use: dataset1, dataset2, or dataset3', selectedDataset);
end

if isempty(datasetPath)
    error('Dataset path is empty! Please set the path for %s in the configuration section.', selectedDataset);
end

fprintf('  Dataset Path: %s\n', datasetPath);
fprintf('  Confidence Threshold: %.2f\n', confThreshold);
fprintf('  IoU Threshold: %.2f\n\n', iouThreshold);

%% Load Dataset
fprintf('Loading image dataset...\n');
try
    imageFiles = load_dataset_images(datasetPath);
    fprintf('Dataset loaded successfully! Found %d images.\n\n', length(imageFiles));
catch ME
    fprintf('Error loading dataset: %s\n', ME.message);
    return;
end

%% Run Detection
fprintf('========================================\n');
fprintf('Starting Drone Detection...\n');
fprintf('========================================\n\n');

try
    % Call the appropriate detection algorithm
    switch lower(selectedAlgorithm)
        case 'yolov4'
            results = yolov4_detector(imageFiles, yolov4ModelPath, ...
                                    confThreshold, iouThreshold, displayResults);
        case 'hog'
            results = hog_detector(imageFiles, hogModelPath, ...
                                  hogThreshold, displayResults);
        case 'point_feature_matching'
            results = point_feature_matching_detector(imageFiles, pointFeatureTemplatePath, ...
                                                     pointFeatureThreshold, displayResults);
        otherwise
            error('Unknown algorithm: %s. Use: yolov4, hog, or point_feature_matching', selectedAlgorithm);
    end
    
    fprintf('\n========================================\n');
    fprintf('Detection Complete!\n');
    fprintf('========================================\n');
    
    % Display results summary
    if isfield(results, 'totalDetections')
        fprintf('Total Drones Detected: %d\n', results.totalDetections);
    end
    if isfield(results, 'processingTime')
        fprintf('Total Processing Time: %.2f seconds\n', results.processingTime);
    end
    if isfield(results, 'numImages')
        fprintf('Images Processed: %d\n', results.numImages);
    end
    if isfield(results, 'avgTimePerImage')
        fprintf('Average Time per Image: %.3f seconds\n', results.avgTimePerImage);
    end
    
    % Save results if requested
    if saveResults
        try
            save(outputResultsPath, 'results');
            fprintf('Results saved to: %s\n', outputResultsPath);
        catch ME
            fprintf('Warning: Could not save results: %s\n', ME.message);
        end
    end
    
catch ME
    fprintf('Error during detection: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  File: %s, Line: %d, Function: %s\n', ...
                ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
    end
end

fprintf('\nProgram completed.\n');
