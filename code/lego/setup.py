import os
import zipfile
import tarfile
import shutil


def move_directory_contents(source_dir: str, target_dir: str):
    # Ensure source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        # Empty the target directory
        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    # Move all contents from source to target
    for item in os.listdir(source_dir):
        source_item_path = os.path.join(source_dir, item)
        target_item_path = os.path.join(target_dir, item)
        shutil.move(source_item_path, target_item_path)

    print(f"All contents moved from '{source_dir}' to '{target_dir}'.")

def unzip_file(zip_path, extract_to):
    """Unzips a .zip file to the specified directory."""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    print(f"Unzipping {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Ldraw unzipped successfully.")
    except Exception as e:
        print(f"Error while unzipping {zip_path}: {e}")

def untar_file(tar_path, extract_to):
    """Extracts a .tar file to the specified directory."""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    print(f"Extracting {tar_path} to {extract_to}...")
    try:
        with tarfile.open(tar_path, 'r') as tar_ref:
            tar_ref.extractall(path=extract_to)
        print("LDCad unzipped successfully.")
    except Exception as e:
        print(f"Error while extracting {tar_path}: {e}")

def move_directory(src, dest):
    """Moves a directory from src to dest."""
    try:
        shutil.move(src, dest)
    except Exception as e:
        print(f"Error while moving {src} to {dest}: {e}")

def setup_ltron_home(ltron_home_path=None):
    ltron_home = "./code/lego/ltron_home"

    if ltron_home_path is not None:
        move_directory_contents(ltron_home, ltron_home_path)

    # Unzip the LDraw files
    if not os.path.exists(f"{ltron_home}/ldraw"):
        complete_file = f"{ltron_home}/complete.zip"
        print("Ldraw not found, unzipping...")
        unzip_file(complete_file, ltron_home)

        # Move the ldraw folder
        move_directory(f"{ltron_home}/complete/ldraw", ltron_home)

    # Extract the LDCad files
    if not os.path.exists(f"{ltron_home}/LDCad-1-6d2-Linux"):
        tar_file = f"{ltron_home}/LDCad-1-6d2-Linux.tar"
        print("LDCad not found, extracting...")
        untar_file(tar_file, ltron_home)

