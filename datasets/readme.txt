Maize Seedling Dataset and Five Experimental Splits

1. Dataset

The complete maize seedling detection dataset is publicly available on Zenodo:
https://doi.org/10.5281/zenodo.23308016

The images and YOLO-format annotations are distributed through Zenodo and are not duplicated in this GitHub repository.

2. Five Original Image-Level Splits

This directory provides the original train/validation/test image lists used in the five repeated random-split experiments reported in the paper.

Archive        Experiment
-------------  ----------
datasets.zip   Run 1
datasets2.zip  Run 2
datasets3.zip  Run 3
datasets4.zip  Run 4
datasets5.zip  Run 5

Each archive contains one folder with three text files:
- train.txt: training images
- val.txt: validation images
- test.txt: test images

Each line in a text file identifies one image. The training, validation, and test subsets were generated at a ratio of 7:2:1 for each run. The split generation did not use a predefined random seed; these archived lists preserve the actual partitions used. All compared models used the same partition within each run. The training seed was fixed at 3407; this is distinct from the split-generation process.

3. How to Use

(a) Download and extract the complete dataset from Zenodo.
(b) Download and extract the required split archive(s) from this directory.
(c) Update the image paths in train.txt, val.txt, and test.txt to match the location of the extracted dataset on your computer, without changing which images belong to each subset.
(d) Point your YOLO dataset configuration to the corresponding train.txt, val.txt, and test.txt lists.

Important: The archived text files retain the absolute Windows image paths from the original experimental environment (for example, D:/demo/corn_detection/yolo_corn/datasets/test/images/...). These paths are not portable and must be adapted to your local dataset directory. The actual directory layout should be checked against the extracted Zenodo files.
