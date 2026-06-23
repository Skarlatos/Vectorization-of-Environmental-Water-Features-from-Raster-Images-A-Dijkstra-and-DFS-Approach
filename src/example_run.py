from pathlib import Path
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.makedirs('raster', exist_ok=True)
os.makedirs('vector', exist_ok=True)

from RVR.rvr import Riverside


def convert_to_single_channel(image):
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, 0]
    raise ValueError('Expected a 3-channel image.')


def preprocess_image(img):
    single_channel_img = convert_to_single_channel(img)
    img_normalized = cv2.normalize(single_channel_img, None, 0, 255, cv2.NORM_MINMAX)
    _, binary_img = cv2.threshold(img_normalized, 50, 255, cv2.THRESH_BINARY)
    return binary_img


def extract_linestrings(binary_image):
    rows, cols = binary_image.shape
    visited = np.zeros_like(binary_image, dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    linestrings = []

    def dfs(r, c, coords):
        stack = [(r, c)]
        while stack:
            x, y = stack.pop()
            coords.append((y, x))
            for dr, dc in directions:
                nr, nc = x + dr, y + dc
                if 0 <= nr < rows and 0 <= nc < cols and binary_image[nr, nc] == 0 and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if binary_image[r, c] == 0 and not visited[r, c]:
                visited[r, c] = True
                coords = []
                dfs(r, c, coords)
                if len(coords) > 1:
                    linestrings.append(LineString(coords))
    return linestrings


def main():
    rvr = Riverside()

    # if rvr.rastermap is None:
    #     raise FileNotFoundError('Raster TIFF not loaded. Check the TIFF path defined in rvr.py.')
    #
    # watermap = rvr.waterize()
    # cv2.imwrite('raster/waterized.png', watermap)

    img = cv2.imread('raster/waterized.png')
    if img is None:
        raise FileNotFoundError('Could not read raster/waterized.png after waterization.')

    binary_img = preprocess_image(img)
    if binary_img is None:
        raise RuntimeError('Binary image generation failed.')

    amazonS1 = (0.0411, -51.0706)
    amazonR1 = (-4.5333, -73.1833)

    start_coord = (amazonS1[1], amazonS1[0])
    goal_coord = (amazonR1[1], amazonR1[0])
    # rvr.watermap = binary_img
    result = rvr.rota(start_coord, goal_coord)

    plt.figure(figsize=(10, 10))
    plt.imshow(rvr.watermap, cmap='gray')
    plt.plot(result[3][0], result[3][1], color='red', linewidth=2)
    plt.scatter([start_coord[0], goal_coord[0]], [start_coord[1], goal_coord[1]],
                c=['blue', 'green'], s=40)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('vector/amazon_dijkstra_route.png', dpi=300)
    plt.show()

    lines = extract_linestrings(binary_img)

    plt.figure(figsize=(10, 10))
    plt.imshow(binary_img, cmap='gray')
    for line in lines:
        x, y = line.xy
        plt.plot(x, y, color='red', linewidth=0.8)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('vector/dfs_all_rivers.png', dpi=300)
    plt.show()

    print('DFS extraction finished.')
    print('Extracted LineStrings:', len(lines))
    print('Saved figure: vector/dfs_all_rivers.png')


if __name__ == '__main__':
    main()
