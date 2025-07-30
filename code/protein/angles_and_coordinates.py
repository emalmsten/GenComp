'''
Code convert xyz coordinate to dihedral angles & bond angles 
https://github.com/microsoft/foldingdiff/blob/main/foldingdiff/angles_and_coords.py
'''
import os
import sys
from typing import *
import functools
import gzip
import warnings
import logging
import glob
from itertools import groupby
import numpy as np
import pandas as pd
from collections import namedtuple, defaultdict
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.sequence import ProteinSequence
sys.path.insert(0, os.getcwd()) 
import protein.Nerf


MINIMAL_ANGLES = ["phi", "psi", "omega"]
EXHAUSTIVE_ANGLES = ["phi", "psi", "omega", "theta_1", "theta_2", "theta_3"]

def coord_to_angles(
    fname: str,
    angles: List[str] = EXHAUSTIVE_ANGLES,
) -> Optional[pd.DataFrame]:
    assert os.path.isfile(fname)
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")
    warnings.filterwarnings("ignore", ".*invalid value encountered in true_div.*")
    #ignore some types of warning
    warnings.filterwarnings("ignore", ".*invalid value encountered in divide.*")
    source = PDBFile.read(fname)
    if source.get_model_count() > 1:
        return None
    source_struct = source.get_structure()[0]
    # Dihedrals angles
    try:
          phi, psi, omega = struc.dihedral_backbone(source_struct) 
          calc_angles = {"phi": phi, "psi": psi, "omega": omega}
    except struc.BadStructureError:
        logging.debug(f"{fname} contains a malformed structure - skipping")
        return None
    
    # Bonds angles
    non_dihedral_angles = [a for a in angles if a not in calc_angles]
    #print(non_dihedral_angles)
    # Gets the N - CA - C for each residue
    backbone_atoms = source_struct[struc.filter_backbone(source_struct)]
    # print('length of the backbone',len(backbone_atoms))
    for a in non_dihedral_angles:
        if a == "theta_1":
            ### tau = N - CA - C internal angles, A change here!!!!!!
            # starting from the first residue!!!
            # Generate an array r starting from 0, taking every 3rd atom to account for N, CA, and C atoms
            r = np.arange(0, len(backbone_atoms), 3)  # Modified to start from 0
            # idx is an array of atom indices, representing the three atoms for each angle. By stacking r, r + 1, and r + 2, N, CA, and C are combined together.
            idx = np.vstack([r, r + 1, r + 2]).T
            #print('idx',idx)
        elif a == "theta_2":  
            # inter-residue angle between two adjacent residues. It measures the angle between the CA and C atoms of one residue and the N atom of the next residue.
            r = np.arange(0, len(backbone_atoms) - 3, 3)
            idx = np.hstack([np.vstack([r + 1, r + 2, r + 3]), np.zeros((3, 1))]).T
        elif a == "theta_3":
            # angle between the C atom of one residue and the N and CA atoms of the next residue.
            r = np.arange(0, len(backbone_atoms) - 3, 3)
            idx = np.hstack([np.vstack([r + 2, r + 3, r + 4]), np.zeros((3, 1))]).T
            
        else:
            raise ValueError(f"Unrecognized angle: {a}")
        calc_angles[a] = struc.index_angle(backbone_atoms, indices=idx.astype(int))

    #for k in calc_angles:
        #print(f"Key: {k}, Length: {len(calc_angles[k])}")

    # Create a DataFrame and fill NaN values with 0
    df = pd.DataFrame({k: calc_angles[k].squeeze() for k in calc_angles})
    df_filled = df.fillna(0)  # Replace NaN with 0
    
    return df_filled

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Gets the angle between u and v"""
    # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    unit_vector = lambda vector: vector / np.linalg.norm(vector)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

SideChainAtomRelative = namedtuple(
    "SideChainAtom", ["name", "element", "bond_dist", "bond_angle", "dihedral_angle"]
)

def collect_aa_sidechain_angles(
    ref_fname: str,
) -> Dict[str, List[SideChainAtomRelative]]:
    """
    Collect the sidechain distances/angles/dihedrals for all amino acids such that
    we can reconstruct an approximate version of them from the backbone coordinates
    and these relative distances/angles/dihedrals

    Returns a dictionary that maps each amino acid residue to a list of SideChainAtom
    objects
    """
    opener = gzip.open if ref_fname.endswith(".gz") else open
    with opener(ref_fname, "rt") as f:
        structure = PDBFile.read(f)
    if structure.get_model_count() > 1:
        raise ValueError
    chain = structure.get_structure()[0]
    retval = defaultdict(list)
    for _, res_atoms in groupby(chain, key=lambda a: a.res_id):
        res_atoms = struc.array(list(res_atoms))
        # Residue name, 3 letter -> 1 letter
        try:
            residue = ProteinSequence.convert_letter_3to1(res_atoms[0].res_name)
        except KeyError:
            logging.warning(
                f"{ref_fname}: Skipping unknown residue {res_atoms[0].res_name}"
            )
            continue
        if residue in retval:
            continue
        backbone_mask = struc.filter_backbone(res_atoms)
        a, b, c = res_atoms[backbone_mask].coord  # Backbone
        for sidechain_atom in res_atoms[~backbone_mask]:
            d = sidechain_atom.coord
            retval[residue].append(
                SideChainAtomRelative(
                    name=sidechain_atom.atom_name,
                    element=sidechain_atom.element,
                    bond_dist=np.linalg.norm(d - c, 2),
                    bond_angle=angle_between(d - c, b - c),
                    dihedral_angle=struc.dihedral(a, b, c, d),
                )
            )
    logging.info(
        "Collected {} amino acid sidechain angles from {}".format(
            len(retval), os.path.abspath(ref_fname)
        )
    )
    return retval


@functools.lru_cache(maxsize=32)
def build_aa_sidechain_dict(
    reference_pdbs: Optional[Collection[str]] = None,
) -> Dict[str, List[SideChainAtomRelative]]:
    """
    Build a dictionary that maps each amino acid residue to a list of SideChainAtom
    that specify how to build out that sidechain's atoms from the backbone
    """
    if not reference_pdbs:
        reference_pdbs = glob.glob(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/*.pdb")
        )

    ref_file_counter = 0
    retval = {}
    for pdb in reference_pdbs:
        try:
            sidechain_angles = collect_aa_sidechain_angles(pdb)
            retval.update(sidechain_angles)  # Overwrites any existing key/value pairs
            ref_file_counter += 1
        except ValueError:
            continue
    logging.info(f"Built sidechain dictionary with {len(retval)} amino acids from {ref_file_counter} files")
    return retval


def add_sidechains_to_backbone(
    backbone_pdb_fname: str,
    aa_seq: str,
    out_fname: str,
    reference_pdbs: Optional[Collection[str]] = None,
) -> str:
    """
    Add the sidechains specified by the amino acid sequence to the backbone
    """
    opener = gzip.open if backbone_pdb_fname.endswith(".gz") else open
    with opener(backbone_pdb_fname, "rt") as f:
        structure = PDBFile.read(f)
    if structure.get_model_count() > 1:
        raise ValueError
    chain = structure.get_structure()[0]

    aa_library = build_aa_sidechain_dict(reference_pdbs)

    atom_idx = 1  # 1-indexed
    full_atoms = []
    for res_aa, (_, backbone_atoms) in zip(
        aa_seq, groupby(chain, key=lambda a: a.res_id)
    ):
        backbone_atoms = struc.array(list(backbone_atoms))
        assert len(backbone_atoms) == 3
        for b in backbone_atoms:
            b.atom_id = atom_idx
            atom_idx += 1
            b.res_name = ProteinSequence.convert_letter_1to3(res_aa)
            full_atoms.append(b)
        # Place each atom in the sidechain
        a, b, c = backbone_atoms.coord
        for rel_atom in aa_library[res_aa]:
            d = Nerf.place_dihedral(
                a,
                b,
                c,
                rel_atom.bond_angle,
                rel_atom.bond_dist,
                rel_atom.dihedral_angle,
            )
            atom = struc.Atom(
                d,
                chain_id=backbone_atoms[0].chain_id,
                res_id=backbone_atoms[0].res_id,
                atom_id=atom_idx,
                res_name=ProteinSequence.convert_letter_1to3(res_aa),
                atom_name=rel_atom.name,
                element=rel_atom.element,
                hetero=backbone_atoms[0].hetero,
            )
            atom_idx += 1
            full_atoms.append(atom)
    sink = PDBFile()
    sink.set_structure(struc.array(full_atoms))
    sink.write(out_fname)
    return out_fname




'''# Check Check
import numpy as np
def calculate_angle(A, B, C):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    BA = A - B
    BC = C - B
    
    dot_product = np.dot(BA, BC)
    
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)
    
    cos_theta = dot_product / (norm_BA * norm_BC)

    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

A = [7.337,  -2.021,  50.761]
B = [7.329,  -2.451,  49.377]
C = [6.228,  -1.644,  48.672]

radian_ABC = np.deg2rad(calculate_angle(A, B, C))
print(radian_ABC)

path = '/mnt/c/Users/chenwanxin/Documents/GitHub/GenComp/cath/dompdb/1a0aA00.pdb'
df = coord_to_angles(path)
#df =  canonical_distances_and_dihedrals(path)
print(df)'''