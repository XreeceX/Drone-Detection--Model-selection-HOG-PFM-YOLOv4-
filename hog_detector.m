function results = hog_detector(imageFiles, modelPath, hogThreshold, displayResults)
%% Histograms of Oriented Gradients (HOG) Algorithm for Drone Detection
% This function performs drone detection using HOG features with SVM classifier
%
% Inputs:
%   imageFiles - Cell array of image file paths
%   modelPath - Path to HOG-SVM model file
%   hogThreshold - Detection threshold
%   displayResults - Boolean to display results
%
% Outputs:
%   results - Structure containing detection results

    fprintf('Initializing HOG detector...\n');
    
    % Initialize results structure
    results = struct();
    results.totalDetections = 0;
    results.detections = [];
    results.processingTime = 0;
    results.numImages = length(imageFiles);
    
    % Load HOG-SVM model
    [hogClassifier, hogParams] = load_hog_model(modelPath);
    
    if isempty(hogClassifier)
        error('Failed to load HOG model');
    end
    
    fprintf('HOG model loaded successfully.\n');
    
    tic; % Start timing
    
    % Process images
    fprintf('Processing %d images with HOG...\n', length(imageFiles));
    results = process_images_hog(hogClassifier, hogParams, imageFiles, hogThreshold, displayResults);
    
    results.processingTime = toc; % End timing
    if results.numImages > 0
        results.avgTimePerImage = results.processingTime / results.numImages;
    end
    
    fprintf('Detection completed in %.2f seconds.\n', results.processingTime);
end

%% Helper Functions

function [classifier, params] = load_hog_model(modelPath)
    % Load HOG-SVM classifier model
    
    fprintf('Loading HOG model from: %s\n', modelPath);
    
    classifier = [];
    params = struct();
    
    % Default HOG parameters
    params.cellSize = [8 8];
    params.blockSize = [2 2];
    params.blockOverlap = [1 1];
    params.numBins = 9;
    
    % Load model if it exists
    if exist(modelPath, 'file')
        try
            load(modelPath, 'classifier', 'hogParams');
            if exist('hogParams', 'var')
                params = hogParams;
            end
            fprintf('HOG model loaded successfully.\n');
        catch ME
            fprintf('Error loading model file: %s\n', ME.message);
            fprintf('Creating placeholder classifier...\n');
            classifier = create_placeholder_hog_classifier();
        end
    else
        fprintf('Model file not found. Creating placeholder...\n');
        fprintf('To use actual HOG detector, train a model and save it to: %s\n', modelPath);
        classifier = create_placeholder_hog_classifier();
    end
end

function classifier = create_placeholder_hog_classifier()
    % Create a placeholder HOG classifier for demonstration
    
    classifier = struct();
    classifier.type = 'hog_svm';
    classifier.isPlaceholder = true;
    
    % In real implementation, you would:
    % 1. Extract HOG features from training images
    % 2. Train an SVM classifier
    % 3. Save the classifier
end

function results = process_images_hog(classifier, params, imageFiles, hogThreshold, displayResults)
    % Process a batch of images using HOG
    
    results = struct();
    results.totalDetections = 0;
    results.detections = [];
    results.numImages = length(imageFiles);
    allDetections = [];
    
    numImages = length(imageFiles);
    
    for i = 1:numImages
        fprintf('Processing image %d/%d: %s\n', i, numImages, imageFiles{i});
        
        try
            % Read image
            img = imread(imageFiles{i});
            
            % Perform detection
            detections = detect_drones_hog(img, classifier, params, hogThreshold);
            
            % Store results
            if ~isempty(detections)
                detections.imageIndex = repmat(i, size(detections, 1), 1);
                detections.imagePath = repmat({imageFiles{i}}, size(detections, 1), 1);
                allDetections = [allDetections; detections];
                results.totalDetections = results.totalDetections + size(detections, 1);
            end
            
            % Display results (optional)
            if displayResults && ~isempty(detections)
                display_detections_hog(img, detections, imageFiles{i});
            end
            
        catch ME
            fprintf('Error processing image %s: %s\n', imageFiles{i}, ME.message);
        end
    end
    
    results.detections = allDetections;
end

function detections = detect_drones_hog(img, classifier, params, threshold)
    % Perform drone detection on a single image using HOG
    
    detections = [];
    
    if isfield(classifier, 'isPlaceholder') && classifier.isPlaceholder
        % Placeholder detection (for demonstration)
        return;
    end
    
    try
        % Convert to grayscale if needed
        if size(img, 3) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end
        
        % Extract HOG features
        hogFeatures = extractHOGFeatures(grayImg, ...
            'CellSize', params.cellSize, ...
            'BlockSize', params.blockSize, ...
            'BlockOverlap', params.blockOverlap, ...
            'NumBins', params.numBins);
        
        % Sliding window detection
        windowSize = [64 64];  % Standard window size for HOG
        stepSize = 16;  % Step size for sliding window
        
        [imgHeight, imgWidth] = size(grayImg);
        detections = [];
        
        for y = 1:stepSize:(imgHeight - windowSize(1) + 1)
            for x = 1:stepSize:(imgWidth - windowSize(2) + 1)
                % Extract window
                window = grayImg(y:(y+windowSize(1)-1), x:(x+windowSize(2)-1));
                
                % Extract HOG features for window
                try
                    windowHOG = extractHOGFeatures(window, ...
                        'CellSize', params.cellSize, ...
                        'BlockSize', params.blockSize, ...
                        'BlockOverlap', params.blockOverlap, ...
                        'NumBins', params.numBins);
                    
                    % Classify using SVM
                    [label, score] = predict(classifier, windowHOG);
                    
                    % Check if drone detected
                    if strcmp(label, 'drone') || (isnumeric(label) && label == 1)
                        if score > threshold || (iscell(label) && score(2) > threshold)
                            % Store detection
                            bbox = [x, y, windowSize(2), windowSize(1)];
                            conf = score;
                            if iscell(conf) || length(conf) > 1
                                conf = conf(2);  % Positive class score
                            end
                            detections = [detections; struct('bbox', bbox, 'score', conf)];
                        end
                    end
                catch
                    % Skip this window if feature extraction fails
                    continue;
                end
            end
        end
        
        % Convert to table format
        if ~isempty(detections)
            bboxes = vertcat(detections.bbox);
            scores = vertcat(detections.score);
            detections = table(bboxes, scores, 'VariableNames', {'bbox', 'Score'});
        end
        
    catch ME
        fprintf('Error in HOG detection: %s\n', ME.message);
    end
end

function annotatedImg = draw_detections_hog(img, detections, imagePath)
    % Draw detection bounding boxes on image
    
    annotatedImg = img;
    
    if isempty(detections)
        return;
    end
    
    for i = 1:size(detections, 1)
        if istable(detections)
            bbox = detections.bbox(i, :);
            score = detections.Score(i);
        else
            bbox = detections(i, 1:4);
            score = detections(i, 5);
        end
        
        label = sprintf('Drone %.2f', score);
        
        % Draw bounding box
        annotatedImg = insertShape(annotatedImg, 'Rectangle', bbox, ...
            'LineWidth', 3, 'Color', 'blue');
        
        % Draw label
        annotatedImg = insertText(annotatedImg, [bbox(1), bbox(2)], label, ...
            'FontSize', 14, 'BoxColor', 'blue', 'BoxOpacity', 0.7);
    end
end

function display_detections_hog(img, detections, imagePath)
    % Display image with detections
    
    annotatedImg = draw_detections_hog(img, detections, imagePath);
    
    figure;
    imshow(annotatedImg);
    title(sprintf('HOG - Detections in: %s', imagePath));
end

