clear; 
clc; 
close all;

I = imread('/Users/moisesgomez/Developer/DIP_FinalProject/DIP-Final-Project/IMG_4951.png');

[centers, radii, labeled] = part1(I, 'extracted-flowers.csv', [0 0.15], [0.17 1], [20 50], 80, 0.98, 1000, [60 3000], [0.95 0.93 0.94]);

% Flowers Found
figure('Name', 'Flowers')
imshow(I, [])
title(sprintf('Number of Flowers: %d', size(radii,1)))
hold on
viscircles(centers, radii, 'LineWidth', 3, 'Color', [0 1 0]);
hold off

figure('Name', 'Labled Mask')
imshow(labeled, [])