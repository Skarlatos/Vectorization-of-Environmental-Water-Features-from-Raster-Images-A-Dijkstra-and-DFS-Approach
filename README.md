This project implements a method for extracting and vectorizing environmental water features from raster images using graph-based algorithms, specifically Dijkstra’s shortest path algorithm and Depth-First Search (DFS).

The goal is to convert raster representations (e.g., satellite or binary water masks) into structured vector paths representing water boundaries or networks.

📌 Overview

The workflow generally follows these steps:

Load raster image (water mask or satellite-derived binary image)
Convert image into a graph representation
Apply DFS to explore connected components
Apply Dijkstra’s algorithm to compute optimal paths along water structures
Generate vectorized output (paths/edges)
📁 Repository Structure
.
├── main.py / script.py        # Main execution script (name may vary)
├── utils/                     # Helper functions for image processing & graph building
├── data/                      # Example raster inputs (if provided)
├── output/                    # Generated vector outputs
└── README.md
⚙️ Requirements

Make sure you have Python 3.8+ installed.

Install dependencies using:

pip install numpy opencv-python matplotlib networkx scipy

Depending on the implementation, you may also need:

pip install scikit-image
🚀 How to Run
Clone the repository:
git clone https://github.com/Skarlatos/Vectorization-of-Environmental-Water-Features-from-Raster-Images-A-Dijkstra-and-DFS-Approach.git
cd Vectorization-of-Environmental-Water-Features-from-Raster-Images-A-Dijkstra-and-DFS-Approach
Run the main script:
python main.py

If the entry file has a different name (e.g., script.py or run.py), replace accordingly.

🧾 Input Format
Input images should be:
Grayscale or binary raster images
Water regions represented as foreground (e.g., white = 1)
Background as 0

Example:

0 0 1 1 0
0 1 1 1 0
0 0 1 0 0
📤 Output

The script produces:

Vectorized representation of water features
Optional visualizations of:
Graph structure
DFS traversal
Dijkstra shortest paths
Output files saved in the output/ directory
🧠 Algorithms Used
DFS (Depth-First Search)

Used to detect connected components in raster water regions.

Dijkstra’s Algorithm

Used to compute optimal paths along water feature graphs based on edge weights.

📊 Example Use Cases
River network extraction
Coastline vectorization
Hydrological modeling
Remote sensing analysis
🛠️ Troubleshooting
Module not found errors → Ensure all dependencies are installed
Image not loading → Check file path and format
No output generated → Verify input is binary or properly thresholded

📦 Data Availability

The raster datasets used in this project are not included in this repository due to their size constraints.

Some input files exceed 25 MB, which is above the file size limit for GitHub repositories. For this reason, the data cannot be uploaded directly.

📩 Access to Data

The data are available upon request.
If you need access, please contact the repository owner.
