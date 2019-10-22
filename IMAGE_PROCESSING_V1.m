%VERSION 1 of Pre-processing code
%specifying the folder where the files live.
myFolder = 'C:\Users\nbtar\Documents\MSE 490\Raw Data';
myNewFolder = 'C:\Users\nbtar\Documents\MSE 490\Processed Images';

%check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end

%get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.tif'); %this is the pattern to recognize the file type
theFiles = dir(filePattern); %counting up the number of files with this pattern

for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);

  %reading the image as an image array with imread()
  imageArray = imread(fullFileName);
  imshow(imageArray);  %display image
 
  %saving the image to a new folder after it has been processed
  set(gcf, 'PaperPositionMode', 'auto'); %this is supposed to not allow the image to shrink but it's still strinking after saving
  saveas(gcf, fullfile(myNewFolder, baseFileName), 'tif'); %saving the image to a new folder
end


