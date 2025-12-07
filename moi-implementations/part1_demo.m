clear; 
clc; 
close all;

I = imread('/Users/moisesgomez/Developer/DIP_FinalProject/DIP-Final-Project/IMG_4951.png');

[centers, radii] = part1(I, [0 0.15], [0.17 1], [20 50], 80, 0.98, 1000, [60 3000], [0.95 0.93 0.94]);

% Flowers Found
figure('Name', 'Flowers')
imshow(I, [])
title(sprintf('Number of Flowers: %d', size(radii,1)))
hold on
viscircles(centers, radii, 'LineWidth', 3, 'Color', [0 1 0]);
hold off

%% Write CSV file

N = size(radii, 1);
mask_numbers = (1:N)';

% Build table
output = [mask_numbers, centers, radii];

% Write to CSV
writematrix(output, 'dip_circles.csv');