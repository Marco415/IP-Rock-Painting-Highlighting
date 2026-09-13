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
1. Find all 8-connected components and identifies the largest one.
2. Identifies anchor objects that are 12% the size of the largest and not on the edge.
3. Keeps components that are large enough and close enough to anchor objects.
4. Objects on margins that don't extend to the center are removed.
5. Small isolated objects are removed.

### Smoothing
1. Closing
2. Hole filling
3. Removes small components
