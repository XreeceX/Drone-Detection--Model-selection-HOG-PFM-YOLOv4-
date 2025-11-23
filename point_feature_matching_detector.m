function results = point_feature_matching_detector(imageFiles, templatePath, threshold, displayResults)
%% Point Feature Matching Algorithm for Drone Detection
% This function performs drone detection using point feature matching (SIFT/SURF/ORB)
%
% Inputs:
%   imageFiles - Cell array of image file paths
%   templatePath - Path to template image or feature file
%   threshold - Matching threshold
%   displayResults - Boolean to display results
%
% Outputs:
%   results - Structure containing detection results

    fprintf('Initializing Point Feature Matching detector...\n');
    
    % Initialize results structure
    results = struct();
    results.totalDetections = 0;
    results.detections = [];
    results.processingTime = 0;
    results.numImages = length(imageFiles);
    
    % Load template features
    [templateImg, templateFeatures, templatePoints] = load_template_features(templatePath);
    
    if isempty(templateFeatures)
        error('Failed to load template features');
    end
    
    fprintf('Template features loaded successfully.\n');
    
    tic; % Start timing
    
    % Process images
    fprintf('Processing %d images with Point Feature Matching...\n', length(imageFiles));
    results = process_images_point_feature(imageFiles, templateImg, templateFeatures, ...
                                          templatePoints, threshold, displayResults);
    
    results.processingTime = toc; % End timing
    if results.numImages > 0
        results.avgTimePerImage = results.processingTime / results.numImages;
    end
    
    fprintf('Detection completed in %.2f seconds.\n', results.processingTime);
end

%% Helper Functions

function [templateImg, features, points] = load_template_features(templatePath)
    % Load template image and extract features
    
    fprintf('Loading template from: %s\n', templatePath);
    
    templateImg = [];
    features = [];
    points = [];
    
    % Check if template file exists
    if exist(templatePath, 'file')
        try
            % Try to load pre-computed features
            if endsWith(templatePath, '.mat')
                load(templatePath, 'templateImg', 'features', 'points');
                if isempty(features)
                    % If features not found, extract from image
                    if ~isempty(templateImg)
                        [features, points] = extract_features(templateImg);
                    end
                end
            else
                % Load template image and extract features
                templateImg = imread(templatePath);
                [features, points] = extract_features(templateImg);
            end
            fprintf('Template features loaded successfully.\n');
        catch ME
            fprintf('Error loading template: %s\n', ME.message);
            fprintf('Creating placeholder template...\n');
            [templateImg, features, points] = create_placeholder_template();
        end
    else
        fprintf('Template file not found. Creating placeholder...\n');
        fprintf('To use actual point feature matching, provide a template image at: %s\n', templatePath);
        [templateImg, features, points] = create_placeholder_template();
    end
end

function [features, points] = extract_features(img)
    % Extract point features from image (SIFT, SURF, or ORB)
    
    % Convert to grayscale if needed
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    
    % Try different feature detectors
    try
        % Try SURF (if available)
        if license('test', 'Computer_Vision_Toolbox')
            points = detectSURFFeatures(grayImg);
            [features, validPoints] = extractFeatures(grayImg, points);
            points = validPoints;
        else
            % Fallback to corner detection
            points = detectHarrisFeatures(grayImg);
            [features, validPoints] = extractFeatures(grayImg, points);
            points = validPoints;
        end
    catch
        % Fallback to basic corner detection
        try
            points = detectHarrisFeatures(grayImg);
            [features, validPoints] = extractFeatures(grayImg, points);
            points = validPoints;
        catch
            % Last resort: create empty features
            features = [];
            points = [];
        end
    end
end

function [templateImg, features, points] = create_placeholder_template()
    % Create a placeholder template for demonstration
    
    % Create a simple placeholder image
    templateImg = zeros(64, 64, 'uint8');
    features = [];
    points = [];
    
    fprintf('Using placeholder template. Please provide a real template image.\n');
end

function results = process_images_point_feature(imageFiles, templateImg, templateFeatures, ...
                                               templatePoints, threshold, displayResults)
    % Process a batch of images using point feature matching
    
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
            detections = detect_drones_point_feature(img, templateImg, templateFeatures, ...
                                                    templatePoints, threshold);
            
            % Store results
            if ~isempty(detections)
                detections.imageIndex = repmat(i, size(detections, 1), 1);
                detections.imagePath = repmat({imageFiles{i}}, size(detections, 1), 1);
                allDetections = [allDetections; detections];
                results.totalDetections = results.totalDetections + size(detections, 1);
            end
            
            % Display results (optional)
            if displayResults
                display_detections_point_feature(img, templateImg, detections, imageFiles{i});
            end
            
        catch ME
            fprintf('Error processing image %s: %s\n', imageFiles{i}, ME.message);
        end
    end
    
    results.detections = allDetections;
end

function detections = detect_drones_point_feature(img, templateImg, templateFeatures, ...
                                                  templatePoints, threshold)
    % Perform drone detection using point feature matching
    
    detections = [];
    
    if isempty(templateFeatures)
        return;
    end
    
    try
        % Convert to grayscale if needed
        if size(img, 3) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end
        
        % Extract features from input image
        [imgFeatures, imgPoints] = extract_features(grayImg);
        
        if isempty(imgFeatures) || isempty(templateFeatures)
            return;
        end
        
        % Match features
        indexPairs = matchFeatures(templateFeatures, imgFeatures, ...
            'MatchThreshold', threshold, 'MaxRatio', 0.6);
        
        if size(indexPairs, 1) < 4
            % Need at least 4 matches for geometric verification
            return;
        end
        
        % Get matched points
        matchedTemplatePoints = templatePoints(indexPairs(:, 1));
        matchedImgPoints = imgPoints(indexPairs(:, 2));
        
        % Estimate geometric transformation
        try
            [tform, inlierTemplatePoints, inlierImgPoints] = ...
                estimateGeometricTransform(matchedTemplatePoints, matchedImgPoints, ...
                'similarity', 'Confidence', 99.9, 'MaxNumTrials', 2000);
            
            if length(inlierTemplatePoints) >= 4
                % Calculate bounding box from template
                [templateHeight, templateWidth] = size(templateImg);
                templateBox = [1, 1, templateWidth, templateHeight];
                
                % Transform template box to image coordinates
                [x, y] = transformPointsForward(tform, ...
                    [templateBox(1), templateBox(1) + templateBox(3)], ...
                    [templateBox(2), templateBox(2) + templateBox(4)]);
                
                % Calculate bounding box
                xMin = min(x);
                xMax = max(x);
                yMin = min(y);
                yMax = max(y);
                
                bbox = [xMin, yMin, xMax - xMin, yMax - yMin];
                
                % Calculate confidence based on number of matches
                numMatches = length(inlierTemplatePoints);
                confidence = min(numMatches / 20, 1.0);  % Normalize to 0-1
                
                % Store detection
                detections = table(bbox, confidence, 'VariableNames', {'bbox', 'Score'});
            end
        catch
            % If geometric transform fails, use simple bounding box from matched points
            if length(matchedImgPoints) >= 4
                locations = matchedImgPoints.Location;
                xMin = min(locations(:, 1));
                xMax = max(locations(:, 1));
                yMin = min(locations(:, 2));
                yMax = max(locations(:, 2));
                
                bbox = [xMin, yMin, xMax - xMin, yMax - yMin];
                confidence = min(size(indexPairs, 1) / 20, 1.0);
                
                detections = table(bbox, confidence, 'VariableNames', {'bbox', 'Score'});
            end
        end
        
    catch ME
        fprintf('Error in point feature matching: %s\n', ME.message);
    end
end

function annotatedImg = draw_detections_point_feature(img, templateImg, detections, imagePath)
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
            'LineWidth', 3, 'Color', 'red');
        
        % Draw label
        annotatedImg = insertText(annotatedImg, [bbox(1), bbox(2)], label, ...
            'FontSize', 14, 'BoxColor', 'red', 'BoxOpacity', 0.7);
    end
end

function display_detections_point_feature(img, templateImg, detections, imagePath)
    % Display image with detections
    
    annotatedImg = draw_detections_point_feature(img, templateImg, detections, imagePath);
    
    figure;
    imshow(annotatedImg);
    title(sprintf('Point Feature Matching - Detections in: %s', imagePath));
end

