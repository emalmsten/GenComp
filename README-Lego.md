
# LTRON Readme

## Getting Started

### Installation

To set up the environment, use the provided `environment.yaml` file as described in the main project README.  
If you plan to visualize outputs, you must also install the following package:

```bash
pip install splendor-render
```

### Running the Program

You can execute the main script using the same steps as in the general README. It should run seamlessly if you are in the project root directory (`GenComp`). If you are not in the project root, specify `root_dir` as an argument when running.

To start training and testing the model, set `train: true` and `test: true` in the configuration file (`code/configs/lego.yaml`).  
All other parameters are defined in the configuration file.

To provide the Weights and Biases (wandb) API key, you can either:
- Add the key directly in the configuration file.
- Provide a path to a file containing the API key in the configuration.

### Visualization

To visualize the output, run the following script with the same arguments as when running the main program:
```bash
python code/lego/visualize_output.py
```
There are some parameters that can be set for the visualization in the aforementioned config file. There are also a few commands you have on the viewer, apart from the ones implemented by the LTRON paper.
These will be printed when running the script. The most important is Q for previewing a gif and G for making a gif.

### Dataset

You can download the dataset from [here](https://github.com/aaronwalsman/ltron).  
It is not needed unless you want more than the random constructions with 4 and 8 bricks, since they are included in the GitHub repository and will be automatically unzipped when you run the main script for the first time.

## Troubleshooting

**OpenGL Context Error**:  
If you encounter an error related to OpenGL stating that there is "no context," update the condition in your code.  
Replace:
```python
if context == 0:
```
With:
```python
if context is None:
``` 
On line 38 of contextdata.py. This will also be displayed at the bottom of the stacktrace.
