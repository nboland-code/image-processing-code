%crack characterization using image analysis
%this is a test for image pre-processing

%first the user must input the number of images that will be read
%prompt = 'Input the number of images to be read\n';     %user must input how many images are to be read
%x = input(prompt);      %x is the variable used to represent the number of images to be read

%we must have a loop to input all the images and go through image
%pre-processing

%loop to input all the images
%for i = 1:x
    %I = imread('C:\Users\nbtar\Documents\UM Winter 2019\MSE 490 W19\Characterizing Cracks\Raw Data\', i);
    %imshow(I);
%end

%to read the image in
I = imread('C:\Users\nbtar\Documents\UM Winter 2019\MSE 490 W19\Characterizing Cracks\Raw Data\500x 1 2 80s.tif');

%the first step is to smoothen the image so that precipitants and grain
%boundaries are less pronounced and the cracks are more pronounced
%J is the variable to be used for the smooth image
J = wiener2(I, [20,20]);     %wiener2 used to smoothen the image
figure;
imshowpair(I, J, 'montage'); %comapring the original image to the smooth image

hold on;

%next, we must adjust the contrast of the image and binarize the image so
%that it is strictly black and white
T = adaptthresh(J, 0.80);    %setting the threshold where a higher number means more contrast
BW = imbinarize(J, T);      %binarizing the threshold and the image 
%we must use Canny method to invert BW in the image so that we can
%dilate the image and begin connecting lines - WE CANNOT DO THIS WITHOUT
%CANNY METHOD
BW2 = edge(BW, 'Canny');    %use Canny method to get rid of the scale bar and invert BW
figure;
imshowpair(BW, BW2, 'montage'); %comparing the binarized and Canny method images

se = strel('line', 5,5);   %creating a vertical line shaped structuring element
BW3 = imdilate(BW2, se);     %dilating the image to connect lines
figure;
imshow(BW3);


