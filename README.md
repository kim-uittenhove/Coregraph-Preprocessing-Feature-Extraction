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
You can apply the pipeline to your own TMT data or use our publicly available dataset available at: [Zenodo link].

## Repository Structure
- `scripts/` : Feature extraction scripts
- `doc/` : PDF documentation of feature extraction method

## License
Distributed under the UNIL–CHUV research-use-only license, which permits usage by academic and research institutions in non-commercial contexts. Commercial use requires prior written consent from UNIL–CHUV (PACTT). - see the LICENSE file for details.

## Acknowledgments
We thank François Beaune (GitHub: @dictoon) for his valuable support in testing and debugging the data acquisition software, for coordinating with the WACOM development team, and for providing insightful feedback on potential algorithmic approaches.
We also thank the Swiss National Science Foundation (SNSF) for supporting this research through the Sinergia grant CRSII15_186239/1, awarded to principal investigators Daniela Jopp, Stefano Cavalli, Armin von Gunten, François Hermann, and Mike Martin.

