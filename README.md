# MDPA
official implementation of the paper MDPA: Margin-based Dual Prototypes Adaptation for Cross-Subject Motor Imagery EEG Decoding
![Graphical Abstract](GraphAbstract.jpg)
## Data download and preprocessing
- get BCI competition IV dataset from https://www.bbci.de/competition/iv/

- get SHU 3C dataset from https://plus.figshare.com/articles/dataset/Brain_Computer_Interface_Motor_Imagery-EEG_Dataset/22671172

a demo eeg preprocess pipeline code is get_raw_data.m

## EEGEncoder
the architecture included in the paper can be found in:
- SST-DPN: https://github.com/hancan16/SST-DPN

## Demo code

This repository provides example implementations for cross-subject motor imagery EEG classification:

- **simple_TDF_demo.py**  
  A minimal demonstration of the proposed Target Domain Focusing (TDF) loss, illustrating its core usage independent of the full MDPA framework.

- **main_2a2b.py**  
  The complete implementation of the MDPA method for cross-subject motor imagery classification on the BCI Competition IV-2a and IV-2b datasets.

- **main_SHU.py**  
  The complete implementation of the MDPA method evaluated on the SHU 3C motor imagery EEG dataset.
