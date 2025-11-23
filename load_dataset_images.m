function imageFiles = load_dataset_images(datasetPath)
%% Load Images from Dataset Folder
% This function loads images from a specified folder
%
% Inputs:
%   datasetPath - Path to folder containing images
%
% Outputs:
%   imageFiles - Cell array of image file paths

    fprintf('Loading images from: %s\n', datasetPath);
    
    if ~exist(datasetPath, 'dir')
        error('Dataset folder does not exist: %s', datasetPath);
    end
    
    % Supported image formats
    imageExtensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'};
    imageFiles = {};
    
    % Load all supported image formats
    for i = 1:length(imageExtensions)
        files = dir(fullfile(datasetPath, imageExtensions{i}));
        if ~isempty(files)
            imageFiles = [imageFiles; cellstr(fullfile(datasetPath, {files.name}'))];
        end
    end
    
    % Also check for uppercase extensions
    for i = 1:length(imageExtensions)
        files = dir(fullfile(datasetPath, upper(imageExtensions{i})));
        if ~isempty(files)
            imageFiles = [imageFiles; cellstr(fullfile(datasetPath, {files.name}'))];
        end
    end
    
    if isempty(imageFiles)
        error('No images found in folder: %s\nSupported formats: JPG, JPEG, PNG, BMP, TIF, TIFF', datasetPath);
    end
    
    fprintf('Found %d images.\n', length(imageFiles));
end

