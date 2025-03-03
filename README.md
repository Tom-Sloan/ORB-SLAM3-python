# ORB-SLAM3-PYTHON
===

Python bindings generated using [pybind11](https://pybind11.readthedocs.io/en/stable/). We use a modified version of ORB-SLAM3 (included as a submodule) to exntend interfaces. It might not be the most up-to-date with the original ORB-SLAM3.

## Update

+ Oct. 3rd, 2023: Added demo code.
+ Feb. 7th, 2023: First working version. 

## Dependancy

+ OpenCV >= 4.4
+ Pangolin
+ Eigen >= 3.1
+ C++11 or C++0x Compiler

## Installation

1. Clone the repo with `--recursive`
2. Install `Eigen`, `Pangolin` and `OpenCV` if you havn't already.
3. `ORB-SLAM3` requires `openssl` to be installed: `sudo apt-get install libssl-dev`
4. Install some pip packages: `pip install pillow pyparsing pytz six watchdog`
5. Run `python setup install` or `pip install .`.
6. Please raise an issue if you run into any.

## Demo

Please see the demo at `demo/run_rgb.py` for how to use this code. For example, you can run this demo with (by substituting the appropriate arguments):

```bash
python demo/run_rgb.py /home/sam3/Desktop/Toms_Workspace/LidarWorld_Server/recordings/ --vocab_file=./Vocabulary/ORBvoc.txt
```
### In docker container
```bash
python3 demo/run_rgbd.py ../data/recordings/20250130_130942 --vocab_file=./third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt
```

```bash
python demo/run_mono_inertial.py \
    --vocab_file=third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt \
    --settings_file=third_party/ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml \
    --sequence_path=/mnt/SSD_4T/EuRoC/MH01/
```

## Note
I found that extracting third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz was not working. I had to download the file from the ORB-SLAM3 github repo and extract it manually.
