
# PuzzleFusion Readme
Implementation extended from the [PuzzleFusion](https://github.com/sepidsh/PuzzleFussion) repository.
## Getting Started

### Installation

To set up the environment, use the provided `environment.yaml` file as described in the main project README.  

### Running the Program

You can execute the main script using the same steps as in the general README. It should run seamlessly if you are in the project root directory (`GenComp`). If you are not in the project root, specify `root_dir` as an argument when running.

To start training and testing the model, set `train: true` and `test: true` in the configuration file (`code/configs/lego.yaml`).  
All other parameters are defined in the configuration file.

To provide the Weights and Biases (wandb) API key, you can either:
- Add the key directly in the configuration file.
- Provide a path to a file containing the API key in the configuration.

### Visualization

The code for generating a sample and visualising the output and diffusion process can be found in `code/puzzlefusion/inference.py`
The model checkpoint and base sample can be changed from the file.
For evaluating with the provided metrics and viewing the three construcred examples, see the python notebook at `code/puzzlefusion/data_vis.ipynb`

### Dataset

The Cross Cut dataset can be accessed via this [crosscut-data](https://drive.google.com/file/d/1kRRI9V6ro1MK0f-rNbw0hg5jw_WVwlzw/view). After unzipping the data folders should follow the structure: `datasets/puzzlefusion/cross_cut`. Move the processed folder into cross_cut so that `datasets/puzzlefusion/cross_cut contains`: `processed`, `train_poly_data`, `test_poly_data`.
