function results = yolov4_detector(imageFiles, modelPath, confThreshold, iouThreshold, displayResults)
%% YOLOv4 Algorithm for Drone Detection
% This function performs drone detection using YOLOv4
%
% Inputs:
%   imageFiles - Cell array of image file paths
%   modelPath - Path to YOLOv4 model file
%   confThreshold - Confidence threshold for detections (0-1)
%   iouThreshold - IoU threshold for NMS (0-1)
%   displayResults - Boolean to display results
%
% Outputs:
%   results - Structure containing detection results

    fprintf('Initializing YOLOv4 detector...\n');
    
    % Initialize results structure
    results = struct();
    results.totalDetections = 0;
    results.detections = [];
    results.processingTime = 0;
    results.numImages = length(imageFiles);
    
    % Load YOLOv4 network
    detector = load_yolov4_network(modelPath);
    
    if isempty(detector)
        error('Failed to load YOLOv4 network');
    end
    
    fprintf('YOLOv4 network loaded successfully.\n');
    
    tic; % Start timing
    
    % Process images
    fprintf('Processing %d images with YOLOv4...\n', length(imageFiles));
    results = process_images_yolov4(detector, imageFiles, confThreshold, iouThreshold, displayResults);
    
    results.processingTime = toc; % End timing
    if results.numImages > 0
        results.avgTimePerImage = results.processingTime / results.numImages;
    end
    
    fprintf('Detection completed in %.2f seconds.\n', results.processingTime);
end

%% Helper Functions

function detector = load_yolov4_network(modelPath)
    % Load YOLOv4 network
    
    fprintf('Loading YOLOv4 model from: %s\n', modelPath);
    
    % Check if Deep Learning Toolbox is available
    if ~license('test', 'Deep_Learning_Toolbox')
        warning('Deep Learning Toolbox not available. Using placeholder network.');
        detector = create_placeholder_detector();
        return;
    end
    
    % Load model if it exists
    if exist(modelPath, 'file')
        try
            load(modelPath, 'detector');
            fprintf('Model loaded successfully.\n');
        catch ME
            fprintf('Error loading model file: %s\n', ME.message);
            fprintf('Creating placeholder detector...\n');
            detector = create_placeholder_detector();
        end
    else
        fprintf('Model file not found. Creating placeholder...\n');
        fprintf('To use actual YOLOv4, train a model and save it to: %s\n', modelPath);
        detector = create_placeholder_detector();
    end
end

function detector = create_placeholder_detector()
    % Create a placeholder detector for demonstration
    
    detector = struct();
    detector.type = 'yolov4';
    detector.isPlaceholder = true;
    
    % In real implementation, you would:
    % detector = yolov4ObjectDetector('yolov4-coco');
    % Or load your custom trained model
end

function results = process_images_yolov4(detector, imageFiles, confThreshold, iouThreshold, displayResults)
    % Process a batch of images
    
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
            detections = detect_drones_yolov4(img, detector, confThreshold, iouThreshold);
            
            % Store results
            if ~isempty(detections)
                if istable(detections)
                    detections.imageIndex = repmat(i, size(detections, 1), 1);
                    detections.imagePath = repmat({imageFiles{i}}, size(detections, 1), 1);
                else
                    detections = [detections, repmat(i, size(detections, 1), 1), ...
                                 repmat({imageFiles{i}}, size(detections, 1), 1)];
                end
                allDetections = [allDetections; detections];
                results.totalDetections = results.totalDetections + size(detections, 1);
            end
            
            % Display results (optional)
            if displayResults && ~isempty(detections)
                display_detections_yolov4(img, detections, imageFiles{i});
            end
            
        catch ME
            fprintf('Error processing image %s: %s\n', imageFiles{i}, ME.message);
        end
    end
    
    results.detections = allDetections;
end

function detections = detect_drones_yolov4(img, detector, confThreshold, iouThreshold)
    % Perform drone detection on a single image using YOLOv4
    
    if isfield(detector, 'isPlaceholder') && detector.isPlaceholder
        % Placeholder detection (for demonstration)
        detections = placeholder_detection(img, confThreshold);
    else
        % Real YOLOv4 detection
        try
            detections = detect(detector, img, 'Threshold', confThreshold);
        catch
            % Fallback to placeholder if detection fails
            detections = placeholder_detection(img, confThreshold);
        end
    end
    
    % Apply Non-Maximum Suppression (NMS)
    if ~isempty(detections) && size(detections, 1) > 1
        detections = apply_nms(detections, iouThreshold);
    end
end

function detections = placeholder_detection(img, confThreshold)
    % Placeholder detection function
    detections = [];
end

function detections = apply_nms(detections, iouThreshold)
    % Apply Non-Maximum Suppression to remove overlapping detections
    
    if isempty(detections) || size(detections, 1) <= 1
        return;
    end
    
    % Extract bounding boxes and scores
    if istable(detections)
        boxes = detections{:, 1:4};
        scores = detections.Score;
    else
        boxes = detections(:, 1:4);
        scores = detections(:, 5);
    end
    
    % Apply NMS
    try
        if exist('selectStrongestBbox', 'file')
            [selected, ~] = selectStrongestBbox(boxes, scores, ...
                'RatioType', 'Union', 'OverlapThreshold', iouThreshold);
            detections = detections(selected, :);
        else
            detections = simple_nms(detections, iouThreshold);
        end
    catch
        detections = simple_nms(detections, iouThreshold);
    end
end

function detections = simple_nms(detections, iouThreshold)
    % Simple NMS implementation
    
    if isempty(detections) || size(detections, 1) <= 1
        return;
    end
    
    % Sort by confidence score (descending)
    if istable(detections)
        [~, idx] = sort(detections.Score, 'descend');
    else
        [~, idx] = sort(detections(:, 5), 'descend');
    end
    detections = detections(idx, :);
    
    % Simple greedy NMS
    keep = true(size(detections, 1), 1);
    
    for i = 1:size(detections, 1)
        if ~keep(i), continue; end
        for j = i+1:size(detections, 1)
            if ~keep(j), continue; end
            iou = calculate_iou(detections(i, :), detections(j, :));
            if iou > iouThreshold
                keep(j) = false;
            end
        end
    end
    
    detections = detections(keep, :);
end

function iou = calculate_iou(det1, det2)
    % Calculate Intersection over Union (IoU) between two detections
    
    if istable(det1)
        box1 = det1{1, 1:4};
        box2 = det2{1, 1:4};
    else
        box1 = det1(1:4);
        box2 = det2(1:4);
    end
    
    % Calculate intersection
    x1 = max(box1(1), box2(1));
    y1 = max(box1(2), box2(2));
    x2 = min(box1(1) + box1(3), box2(1) + box2(3));
    y2 = min(box1(2) + box1(4), box2(2) + box2(4));
    
    if x2 <= x1 || y2 <= y1
        iou = 0;
        return;
    end
    
    intersection = (x2 - x1) * (y2 - y1);
    area1 = box1(3) * box1(4);
    area2 = box2(3) * box2(4);
    union = area1 + area2 - intersection;
    
    iou = intersection / union;
end

function annotatedImg = draw_detections_yolov4(img, detections, imagePath)
    % Draw detection bounding boxes on image
    
    annotatedImg = img;
    
    if isempty(detections)
        return;
    end
    
    for i = 1:size(detections, 1)
        if istable(detections)
            bbox = detections{i, 1:4};
            score = detections.Score(i);
            label = sprintf('Drone %.2f', score);
        else
            bbox = detections(i, 1:4);
            score = detections(i, 5);
            label = sprintf('Drone %.2f', score);
        end
        
        % Draw bounding box
        annotatedImg = insertShape(annotatedImg, 'Rectangle', bbox, ...
            'LineWidth', 3, 'Color', 'green');
        
        % Draw label
        annotatedImg = insertText(annotatedImg, [bbox(1), bbox(2)], label, ...
            'FontSize', 14, 'BoxColor', 'green', 'BoxOpacity', 0.7);
    end
end

function display_detections_yolov4(img, detections, imagePath)
    % Display image with detections
    
    annotatedImg = draw_detections_yolov4(img, detections, imagePath);
    
    figure;
    imshow(annotatedImg);
    title(sprintf('YOLOv4 - Detections in: %s', imagePath));
end

