# Trail Making Test Analysis
Feature extraction code accompanying the paper: *"Dual-Component Analysis of Trail Making Test: Task vs. Movement Coordination "*.

## Overview
This repository contains Python scripts for extracting cognitive task and movement coordination features from kinematic data recorded with a digital pen tablet during the Trail Making Test (TMT). 
The code is organized into sequential analysis steps, each building on the results of the previous step.

## Installation
Clone the repository:
git clone https://github.com/kim-uittenhove/Coregraph-Preprocessing-Feature-Extraction.git
cd your-repo

Install dependencies:
pip install -r requirements.txt

## Usage
Refer to the PDF documentation for a detailed guide to the analysis pipeline. Each script is designed to produce output that serves as input for the next step.
You can apply the pipeline to your own TMT data or use our publicly available dataset.

Uittenhove, K., Jopp, D., Von Gunten, A., & Richiardi, J. (2025). Cognitive and Graphomotor Data from Young Adults [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16753616

## Repository Structure
- `scripts/` : Feature extraction scripts
- `doc/` : PDF documentation of feature extraction method

## License
This code is released under the UNIL–CHUV Software License Agreement for Academic Non-Commercial Research Purposes Only. See the [LICENSE.txt](LICENSE.txt) file for the complete terms and conditions.

## Citation
Uittenhove, K., Richiardi, J., Von Gunten, A., & Jopp, D. (2025). Feature Extraction Code for Digital Pen Tablet Data (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.16753880

[![DOI](https://zenodo.org/badge/1011183959.svg)](https://doi.org/10.5281/zenodo.16534675)

## Acknowledgments
We thank François Beaune (GitHub: @dictoon) for his valuable support in testing and debugging the data acquisition software, for coordinating with the WACOM development team, and for providing insightful feedback on potential algorithmic approaches.

## Funding Sources
The Swiss National Science Foundation (SNSF) supported this research through the Sinergia grant CRSII15_186239/1, awarded to principal investigators Daniela Jopp, Stefano Cavalli, Armin von Gunten, François Hermann, and Mike Martin.

The Faculty of Social and Political Sciences (SSP) at the University of Lausanne provided additional funding for acquisition of hardware and data collection.

Lausanne University Hospital (CHUV), through the Department of Old Age Psychiatry (SUPAA), funded the development of the acquisition software and supported the completion phase of the project.


