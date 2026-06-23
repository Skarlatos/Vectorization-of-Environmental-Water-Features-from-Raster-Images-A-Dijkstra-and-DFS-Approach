# Vectorization of Environmental Water Features from Raster Images: A Dijkstra and DFS Approach

A Python project for vectorizing environmental water features from raster images using graph-based traversal methods, including Dijkstra’s algorithm and Depth-First Search (DFS).

## Project Structure

```text
.
├── raster/
├── src/
├── LICENSE
├── README.md
└── requirements.txt
```

## Files

- `src/` — source code for the vectorization workflow.
- `raster/` — input raster data and/or generated raster examples.
- `example_run.py` — example script as plug and play example.
- `rvr.py` — main module for raster vectorization routines.
- `requirements.txt` — Python dependencies.
- `LICENSE` — license for the project.

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

If the project uses a virtual environment, you can create one first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the example script to process a raster and generate vector output:

```bash
python example_run.py
```

If you want to use the main module directly:

```bash
python rvr.py
```

## Description

This repository implements a raster-to-vector workflow for detecting and tracing water features from image data. The approach combines shortest-path logic and depth-first traversal to follow connected structures and produce vector-like representations of environmental water bodies.

## Input Data

Place raster images inside the `raster/` folder or update the script paths to point to your local data.

## Output

The project is expected to produce vectorized representations of water features, which may be saved as:

- polylines,
- GIS-compatible files, depending on the implementation.

## License

See the `LICENSE` file for license details.
