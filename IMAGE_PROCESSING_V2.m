%---- this code pre-processes all the pictures in myFolder and saves them
%to myNewFolder ----%

%specifying the folder where the files live
myFolder = 'C:\Users\nbtar\Documents\MSE 490\Raw Data';
%specifying the folder where the files will be saved
myNewFolder = 'C:\Users\nbtar\Documents\MSE 490\Processed Images';

%check to make sure that folder actually exists and warn the user if it doesn't
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end

%get a list of all files in the folder with the desired file name pattern
filePattern = fullfile(myFolder, '*.tif'); %this is the pattern to recognize the file type
%counting up the number of files with this pattern
theFiles = dir(filePattern); 

for k = 1 : length(theFiles)
    I = theFiles(k).name;
    fullFileName = fullfile(myFolder, I);
    fprintf(1, 'Now reading %s\n', fullFileName);
    
    %reading the image as an image array with imread()
    imageArray = imread(fullFileName);
         
    %the first step is to smoothen the image so that precipitants and grain
    %boundaries are less pronounced and the cracks are more pronounced
    %J is the variable to be used for the smooth image
    J = wiener2(imageArray, [20,20]);     %wiener2 used to smoothen the image
    
    %next, we must adjust the contrast of the image and binarize the image so
    %that it is strictly black and white
    T = adaptthresh(J, 0.80);    %setting the threshold where a higher number means more contrast
    BW = imbinarize(J, T);      %binarizing the threshold and the image 
    %we must use Canny method to invert BW in the image so that we can
    %dilate the image and begin connecting lines - WE CANNOT DO THIS WITHOUT
    %CANNY METHOD
    BW2 = edge(BW, 'Canny');    %use Canny method to get rid of the scale bar and invert BW

    se = strel('line', 5,5);   %creating a vertical line shaped structuring element
    BW3 = imdilate(BW2, se);     %dilating the image to connect lines
    %imshow(BW3);    %showing the resultant processed images if desired

    %saving the image to a new folder after it has been processed
    %this is supposed to not allow the image to shrink but it's still
    %shrinking after saving?
    set(gcf, 'PaperPositionMode', 'auto'); 
    saveas(gcf, fullfile(myNewFolder, I), 'tif'); %saving the image to a new folder   
end
