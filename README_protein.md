Original Code from: https://github.com/microsoft/foldingdiff

## Download the data to datasets folder and split into train, test and val
#### cd into data directory and execute shell script
wget -P cath ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz

#### cd into the cath directory and untar the file

cd datasets  

cd cath

tar -xzf cath-dataset-nonredundant-S40.pdb.tgz

#### Unify the files to pdb format
for file in *; do mv "$file" "$file.pdb"; done

#### Create train, test, and val folders
mkdir -p train test val

#### Get all .pdb files and sort them alphabetically
files=($(ls *.pdb | sort))

#### Calculate total number of files and split proportions
total_files=${#files[@]}

train_count=$((total_files * 80 / 100))

val_count=$((total_files * 10 / 100))

test_count=$((total_files - train_count - val_count))

#### Distribute files in sequence into each folder
for i in "${!files[@]}"; do
    if (( i < train_count )); then
        mv "${files[$i]}" train/
    elif (( i < train_count + val_count )); then
        mv "${files[$i]}" val/
    else
        mv "${files[$i]}" test/
    fi
done

## Run the main.py file 
Follow the installation readme. Train the model by changing config_dir="./code/configs/protein.yaml", root_dir="your/path/to/ root"
The experiment results will be saved in the folder "./experiments/protein"

## Generate protein
By running the sample.py, replace the path to the checkpoint
You can choose the range of length and the number of proteins you want to generate, and whether save the intermediates.

## (Optional) Transfer the generated protein from.npy to pdb files 
Using the to_pdb.py for further evaluation. Only do this when necessary since the pdb files with intermediate steps cost a lot of storage

## Evaluation
All the evaluation are done in jupyter notebook in evaluation.ipynb which includes the following parts: 
###### Visualization of sampled protein
###### Matching the angles distribution 
###### Ramanchandran plots
###### Results of evaluation
