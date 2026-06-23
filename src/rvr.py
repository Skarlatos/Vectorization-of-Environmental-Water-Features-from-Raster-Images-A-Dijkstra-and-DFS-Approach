# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:34:43 2023

@author: SkaR
"""
import os
from skimage.graph import route_through_array
from skimage.measure import label, regionprops
from scipy.ndimage import rotate
import random
import pandas as pd
from sys import platform
import numpy as np
from skimage.morphology import binary_closing, disk
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
from shapely import dwithin, LineString, MultiLineString
from geopy.distance import geodesic
import geopandas as gpd


# GRAY_HR_SR_W.tif do not contain valued information for rivers
# HYP_HR_SR_OB_DR.tif has valid information for rivers and drains in general
class Riverside:
    def __init__(self):
        """
        Initialized class attributes

        Returns
        -------
        Class attributes.

        """
        if platform == "linux" or platform == "linux2":
            self.mapath = 'raster'
        else:
            self.mapath = 'Topo'
        self.rastermap = cv2.imread(f'{self.mapath}/HYP_HR_SR_OB_DR/HYP_HR_SR_OB_DR.tif')
        if os.path.isfile('raster/waterized.png'):
            self.watermap = cv2.imread('raster/waterized.png')
        else:
            self.watermap = None
        self.originX = -180
        self.originY = 90.00000000000001
        self.pixelWidth = (self.originY * 2) / self.rastermap.shape[0]
        self.pixelHeight = (self.originX * 2) / self.rastermap.shape[1]

    def waterize(self):
        self.watermap = copy.copy(self.rastermap)
        for px in range(self.watermap.shape[0]):
            for py in range(self.watermap.shape[1]):
                # blue da be da
                if np.amax(np.array([self.watermap[px][py][0], self.watermap[px][py][1], self.watermap[px][py][2]])) == \
                        self.watermap[px][py][0]:
                    ### be carefule with icy areas (icy gray includes max blue)
                    if self.watermap[px][py][0] - self.watermap[px][py][2] > 25 or self.watermap[px][py][0] - \
                            self.watermap[px][py][1] > 25:
                        self.watermap[px][py][0] = 255
                        self.watermap[px][py][1] = 0
                        self.watermap[px][py][2] = 0
                    else:
                        pass
                else:
                    pass
        self.watermap = cv2.cvtColor(self.watermap, cv2.COLOR_BGR2GRAY)
        return self.watermap

    def difference(self):
        # Load the two images
        # image1 = cv2.imread(f'{self.mapath}/HYP_HR_SR_OB_DR/HYP_HR_SR_OB_DR.tif')
        image1 = cv2.imread(f'{self.mapath}/HYP_HR_SR_W_DR/HYP_HR_SR_W_DR.tif')
        image2 = cv2.imread(f'{self.mapath}/HYP_HR_SR_W/HYP_HR_SR_W.tif')        # Convert images to grayscale
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between the two images
        difference = cv2.absdiff(image1_gray, image2_gray)

        # Apply a low threshold to the difference image to highlight minor differences
        threshold_value = 10  # Lower value for high sensitivity
        _, binary_difference = cv2.threshold(difference, threshold_value, 255, cv2.THRESH_BINARY)

        # Create a color difference image by applying a colormap
        color_difference = cv2.applyColorMap(binary_difference, cv2.COLORMAP_JET)

        # Overlay the color difference on the original image
        overlay = cv2.addWeighted(image1, 0.7, color_difference, 0.3, 0)
        plt.figure()
        plt.imshow(binary_difference)
        plt.savefig(f'{self.mapath}/binary_difference.eps', format='eps')
        plt.show()

        plt.figure()
        plt.imshow(color_difference)
        plt.savefig(f'{self.mapath}/color_difference.eps', format='eps')
        plt.show()
        return binary_difference


    def coord2pixelOffset(self, img, originX, originY, x, y):
        """
        Parameters
        ----------
        img : numpy.array or image file (png, jpeg, etc..)
            Earth map.
        originX : integer
            The coordinate which corresponds to the column index (i.e the first y coord. of the first cell (0, 0) is -180.
        originY : integer
            The coordinate which corresponds to the column index (i.e the first y coord. of the first cell (0, 0) is 90.
        x : float
            The latitude (in decimal degrees) to be converted (should be between -90 and 90).
        y : float
            The longitude (in decimal degrees) to be converted (should be between -180 and 180).

        Returns
        -------
        xOffset : integer
            The converted coord to array index (x-axis).
        yOffset : integer
            The converted coord to array index (y-axis).

        """
        # by default the georeferencing origin at 0,0 array point (image) is -180, 90.00000000000001
        shape = img.shape
        # pixelWidth =  0.01666666666667
        # pixelHeight =  -0.01666666666667
        # pixelWidth =  (originY*2)/shape[0]
        # pixelHeight = (originX*2)/shape[1]
        pixelWidth = (originY * shape[1] / shape[0]) / shape[0]
        pixelHeight = (originX * shape[1] / shape[0]) / shape[1]
        xOffset = int((y - originX) / pixelWidth)
        yOffset = int((x - originY) / pixelHeight)
        return xOffset, yOffset

    def pixel2coordOffset(self, indices):
        idx_coord = indices[:]
        idx_coord[1] = idx_coord[1] * self.pixelWidth + self.originX
        idx_coord[0] = idx_coord[0] * self.pixelHeight + self.originY
        return idx_coord

    def createPath(self, startCoord, stopCoord, mode):
        # coordinates to array index
        startCoordX = startCoord[0]
        startCoordY = startCoord[1]

        stopCoordX = stopCoord[0]
        stopCoordY = stopCoord[1]

        if mode == 'rl':
            # startIndexX, startIndexY = self.coord2pixelOffset(self.watermap, startCoordX, startCoordY)
            # stopIndexX, stopIndexY = self.coord2pixelOffset(self.watermap, stopCoordX, stopCoordY)
            startIndexX, startIndexY = self.coord2pixelOffset(
                self.watermap, self.originX, self.originY, startCoordX, startCoordY
            )

            stopIndexX, stopIndexY = self.coord2pixelOffset(
                self.watermap, self.originX, self.originY, stopCoordX, stopCoordY
            )
        elif mode == 'gc':
            startIndexX, startIndexY = startCoord[0], startCoord[1]
            stopIndexX, stopIndexY = stopCoord[0], stopCoord[1]
        else:
            print('plz input values ''rl'' or ''gc''')
        # create path(rhumpline)
        path_idx, weight = route_through_array(self.watermap, (startIndexY, startIndexX), (stopIndexY, stopIndexX),
                                               geometric=True, fully_connected=True)
        path_idx = np.array(path_idx).T
        path_idx = path_idx.astype(float)

        return path_idx

    def antimeridian_coords(y, x):
        if y >= 0 + 1 ** (-6) and y <= 180 - 1 ** (-6):
            geo_y = y - 180
            geo_x = x
        elif y >= -180 + 1 ** (-6) and y <= 0 - 1 ** (-6):
            geo_y = y + 180
            geo_x = x
        else:
            print("Mess with the best die like the rest, try new coordinates except 0.0.. and 180.00..")
        return geo_y, geo_x

    def rota(self, CordS, CordR):
        plt.clf()

        # S = []
        # R = []
        CordSinv = (CordS[1], CordS[0])
        CordRinv = (CordR[1], CordR[0])

        print('Go for rl')
        if self.watermap is None and os.path.isfile('raster/waterized.png'):
            print('Waterizing...')
            self.waterize()
            cv2.imwrite('raster/waterized.png', self.watermap)
        else:
            print('Already converted, skip waterizing...')
            self.watermap = cv2.imread('raster/waterized.png')[:, :, 0]
        # self.watermap = self.watermap[:,:,0]
        # print(CordSinv, CordRinv)
        indexes = self.createPath(CordSinv, CordRinv, 'rl')

        # # plt.scatter(indexes[1], indexes[0], linestyle='-', linewidth = 0.05, marker = 'x', color='r', s = 0.1)
        # # for i in range(len([indexes[0]])):
        # #     plt.scatter(indexes[i][0], indexes[i][1],linestyle='-', linewidth = 1, marker = 'x', color='r', s = 0.1)
        plt.plot(indexes[1], indexes[0], linewidth=1, color='r')
        plt.imshow(self.watermap)
        plt.title('Vectorized')
        plt.savefig(f'vector/River_{CordS}_{CordR}.png', dpi=300)
        plt.show()
        # print('IDX1: ',indexes)
        # print('IDX1.1: ',indexes)
        idx_out = indexes[:]
        indices = self.pixel2coordOffset(indexes)
        # print('IDX2: ',idx_out)
        return [CordS, CordR, indices, idx_out]

    def pixeline(self, coords_y, coords_x):
        return LineString(list(zip(coords_y, coords_x)))

    def calculate_line_length(self, coordinates):
        total_length = 0.0
        for i in range(len(coordinates) - 1):
            # Calculate distance between consecutive points
            distance = geodesic(coordinates[i], coordinates[i + 1]).km  # in meters
            total_length += distance

        return total_length


# def preprocess_image(img):
#     # Convert to single channel
#     single_channel_img = convert_to_single_channel(img)
#
#     # Normalize the image to range 0-255
#     img_normalized = cv2.normalize(single_channel_img, None, 0, 255, cv2.NORM_MINMAX)
#
#     # Convert to binary image (make water black and land white)
#     _, binary_img = cv2.threshold(img_normalized, 50, 255, cv2.THRESH_BINARY)
#
#     return binary_img
#
#
# def process_image(binary_image, window_size, black_threshold):
#     # Get the dimensions of the image
#     height, width = binary_image.shape
#
#     # Create a copy of the image to modify
#     modified_image = binary_image.copy()
#
#     # Iterate through the image with a rolling window of defined size
#     # for y in range(0, height, window_size):
#     #     for x in range(0, width, window_size):  # Move one column each time
#     for y in range(0, height, 1):
#         for x in range(0, width, 1):  # Move one column each time
#             # Define the window boundaries
#             x_end = min(x + window_size, width)
#             y_end = min(y + window_size, height)
#
#             # Extract the window
#             window = binary_image[y:y_end, x:x_end]
#
#             # Count the number of black pixels in the window
#             black_pixels_count = np.sum(window == 0)
#
#             # If the count of black pixels exceeds the threshold, set the window to white
#             if black_pixels_count >= black_threshold:
#                 modified_image[y:y_end, x:x_end] = 255
#
#     return modified_image
#
#
# def detect_edges(filtered_img):
#     # Apply Canny edge detection
#     edges = cv2.Canny(filtered_img, 100, 200)
#     return edges
#
# rvr = Riverside()

# cntr_img = rvr.get_poly('China')
#
# start_cntr = (cntr_img[0], cntr_img[1])
# goal_cntr = (cntr_img[2], cntr_img[3])
# originX = -180
# originY = 90.00000000000001
#
# start_cn_px = rvr.coord2pixelOffset(rvr.watermap, originX, originY, start_cntr[1], start_cntr[0])
# goal_cn_px = rvr.coord2pixelOffset(rvr.watermap, originX, originY, goal_cntr[1], goal_cntr[0])
# # start_cn_px = (start_cn_px[0], start_cn_px[1])
# # goal_cn_px = (goal_cn_px[0], goal_cn_px[1])
#
# croped_surface = rvr.watermap[min(start_cn_px[1], goal_cn_px[1]):max(start_cn_px[1], goal_cn_px[1]),
#                  min(start_cn_px[0], goal_cn_px[0]):max(start_cn_px[0], goal_cn_px[0])]
# binary_img = preprocess_image(croped_surface)
#
# binary_img = preprocess_image(rvr.watermap)
#
#
# # Load binary image as grayscale
#
# # Ensure the image was loaded successfully
# if binary_img is None:
#     print("Error: Could not read the image.")
#     exit()
#
#
# binary_difference = rvr.difference()


##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
# import matplotlib
# matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg', etc.
# # # Define window size and black pixel threshold
# # window_size = int(min(binary_img.shape)*0.1)  # Example window size (adjust as needed)
# window_size = 10  # Example window size (adjust as needed)
#
# black_threshold = window_size * window_size * 0.995  # Example threshold (90% black pixels)
# # Process the image
# modified_image = process_image(binary_img, window_size, black_threshold)

# def percentage_difference_2d(array1, array2):
#     # Check if the arrays have the same dimensions
#     if len(array1) != len(array2) or any(len(row1) != len(row2) for row1, row2 in zip(array1, array2)):
#         raise ValueError("Arrays must have the same dimensions")
#
#     total_elements = len(array1) * len(array1[0])
#     differences = 0
#
#     # Iterate through each element and count differences
#     for i in range(len(array1)):
#         for j in range(len(array1[i])):
#             if array1[i][j] != array2[i][j]:
#                 differences += 1
#
#     # Calculate the percentage of differences
#     percentage = (differences / total_elements) * 100
#
#     return percentage
#
# binary_inv = cv2.bitwise_not(binary_difference)
#
# percentage_diff = percentage_difference_2d(binary_inv, modified_image)
# print(f"The percentage of differing values is: {percentage_diff:.2f}%")
#
# def extract_linestrings(binary_image):
#     rows, cols = binary_image.shape
#     linestrings = []
#
#     # Directions for 8-connectivity
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
#
#     visited = np.zeros_like(binary_image, dtype=bool)
#
#     def dfs(r, c, coords):
#         stack = [(r, c)]
#         while stack:
#             x, y = stack.pop()
#             coords.append((y, x))  # Append in (column, row) format
#             for dr, dc in directions:
#                 nr, nc = x + dr, y + dc
#                 if 0 <= nr < rows and 0 <= nc < cols and binary_image[nr, nc] == 0 and not visited[nr, nc]:
#                     visited[nr, nc] = True
#                     stack.append((nr, nc))
#
#     for r in range(rows):
#         for c in range(cols):
#             if binary_image[r, c] == 0 and not visited[r, c]:
#                 visited[r, c] = True
#                 coords = []
#                 dfs(r, c, coords)
#                 # Only create a LineString if there are more than one point
#                 if len(coords) > 1:
#                     linestrings.append(LineString(coords))
#
#     return linestrings
#
#
#
#
# vectorized = extract_linestrings(modified_image)
#
# # Plotting
# plt.figure(figsize=(10, 10))
# plt.imshow(modified_image, cmap='gray')  # Show the original image
#
# # Overlay the LineStrings
# for line in vectorized:
#     x, y = line.xy  # Get the coordinates
#     plt.plot(x, y, color='red', linewidth=2)  # Plot the lines
#
# plt.axis('off')  # Hide axes
# plt.show()
#
#
#
#
#
#
# import numpy as np
# from shapely.geometry import LineString
#
# def linestrings_to_mask(linestrings, shape):
#     mask = np.ones(shape, dtype=np.uint8)  # background = 1
#     rows, cols = shape
#
#     for line in linestrings:
#         coords = np.array(line.coords)
#         for x, y in coords:
#             r = int(round(y))
#             c = int(round(x))
#             if 0 <= r < rows and 0 <= c < cols:
#                 mask[r, c] = 0  # river = 0
#     return mask
# import cv2
# import numpy as np
#
# def linestrings_to_mask_thick(linestrings, shape, thickness=1):
#     mask = np.ones(shape, dtype=np.uint8)
#     for line in linestrings:
#         pts = np.array([(int(round(x)), int(round(y))) for x, y in line.coords], dtype=np.int32)
#         if len(pts) > 1:
#             cv2.polylines(mask, [pts], isClosed=False, color=0, thickness=thickness)
#     return mask
#
#
# pred_mask = linestrings_to_mask_thick(vectorized, binary_img.shape, thickness=2)
# PD = percentage_difference_2d(binary_img, pred_mask)
