# IP-Rock-Painting-Highlighting
Image processing script that segments and saturates rock paintings

### Segmentation
1. Extract A and B channels from LAB image, and saturation from HSV image
2. Determine how much a pixel looks like the desired colour pigment and strengthen those pixels.
3. Strong Gaussian is subtracted from original to remove background variation.
4. Median and weaker Gaussian removes small texture noise.
5. Thresholds identifies strong seed and acceptable seed.
6. 8-connected components expand from strong seeds into acceptable seeds.
7. Density check removes isolated pixels.
8. Small components are removed.

### Mask cleaning
1. Extract A and B channels, and calculate average channel values in mask.
2. Remove pixels that differ too much from the average.
3. Perform density check and remove pixels that are too isolated.
4. Identify and keep 5 largest connected components.
5. Threshold keeps larger components and nearby smaller components.
6. Median filter smooths pixelated edges.

### Smoothing
1. Closing
2. Hole filling
3. Removes small components
