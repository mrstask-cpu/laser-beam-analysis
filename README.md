# Laser Beam Analysis

Python pipeline for quantitative analysis of laser beam profile images:
image preprocessing, beam-axis detection, inverse Abel reconstruction
(BASEX + Hansen–Law), radial Gaussian fitting, beam propagation analysis
and QA/QC reporting.

## Status

Work in progress — v1.0 packaging of an existing research prototype.

## Pipeline

    Raw image
        ↓
    Camera corrections / calibration
        ↓
    Beam-axis detection & alignment
        ↓
    Background / noise preprocessing
        ↓
    Abel center detection
        ↓
    Inverse Abel (BASEX + Hansen–Law)
        ↓
    Radial profiles
        ↓
    Gaussian fitting
        ↓
    Beam propagation / waist
        ↓
    Validation & QA report

## Installation

    py -3.13 -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt

## Usage

TBD (pipeline entry point will be added).

## License

TBD