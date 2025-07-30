import os
import numpy as np
from protein.Nerf import NERFBuilder

input_folder = "code/protein/intermediates"### Change to your actual path to folder
output_folder = "code/protein/intermediates_pdb"  

os.makedirs(output_folder, exist_ok=True)

# all length
for length_folder in os.listdir(input_folder):
    length_path = os.path.join(input_folder, length_folder)
    if os.path.isdir(length_path):
        # sample dir
        for sample_folder in os.listdir(length_path):
            sample_path = os.path.join(length_path, sample_folder)
            if os.path.isdir(sample_path):
                #  .npy fils
                for sample_file in os.listdir(sample_path):
                    if sample_file.endswith(".npy"):
                        file_path = os.path.join(sample_path, sample_file)
                        
                        try:
                            # reshape data
                            output = np.load(file_path)
                            reshaped_output = output.T  # (6, N) --> (N, 6)
                            
                            #  phi, psi, omega 
                            angles = reshaped_output[:, :3]
                            phi = angles[:, 0] * np.pi  
                            psi = angles[:, 1] * np.pi  
                            omega = angles[:, 2] * np.pi  
                     
                            nerf_builder = NERFBuilder(phi, psi, omega)
                            coordinate = nerf_builder.cartesian_coords
                            
  
                            pdb_str = ""
                            atom_names = ['N', 'CA', 'C']
                            for i in range(0, len(coordinate), 3):
                                for j in range(3):
                                    pdb_str += f"ATOM  {i+j+1:5d}  {atom_names[j]:<2}  ALA A{i//3+1:4d}    {coordinate[i+j][0]:8.3f}{coordinate[i+j][1]:8.3f}{coordinate[i+j][2]:8.3f}  1.00  0.00           {atom_names[j][0]}  \n"
                            

                            pdb_filename = f"{sample_file.split('.')[0]}.pdb"
                            pdb_output_path = os.path.join(output_folder, length_folder, sample_folder)
                            os.makedirs(pdb_output_path, exist_ok=True)
                            pdb_output_file = os.path.join(pdb_output_path, pdb_filename)

                            with open(pdb_output_file, "w") as pdb_file:
                                pdb_file.write(pdb_str)

                            print(f"PDB saved as: {pdb_output_file}")

                        except Exception as e:
                            print(f"wrong {file_path} : {e}")

print("all .npy files are transformed to pdb files")
