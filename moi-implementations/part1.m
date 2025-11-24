clear; clc; close all;

I = imread('IMG_4951.png');

R = I(:,:, 1);
G = I(:,:, 2);
B = I(:,:, 3);

RRR = G-B;
RRR = RRR<30;

% Find circles
[centers, radii] = imfindcircles(RRR, [15 90], "Sensitivity", 0.80);

% Overlay detected circles
imshow(I)
hold on
viscircles(centers, radii, 'LineWidth', 3);