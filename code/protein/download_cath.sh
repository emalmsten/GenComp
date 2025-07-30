# cd into data directory and execute shell script
wget -P cath ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz

# cd into the cath directory and untar the file

cd datasets
cd cath
tar -xzf cath-dataset-nonredundant-S40.pdb.tgz

for file in *; do mv "$file" "$file.pdb"; done
# Create train, test, and val folders
mkdir -p train test val

# Get all .pdb files and sort them alphabetically
files=($(ls *.pdb | sort))

# Calculate total number of files and split proportions
total_files=${#files[@]}
train_count=$((total_files * 80 / 100))
val_count=$((total_files * 10 / 100))
test_count=$((total_files - train_count - val_count))

# Distribute files in sequence into each folder
for i in "${!files[@]}"; do
    if (( i < train_count )); then
        mv "${files[$i]}" train/
    elif (( i < train_count + val_count )); then
        mv "${files[$i]}" val/
    else
        mv "${files[$i]}" test/
    fi
done