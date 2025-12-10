import math
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from rdkit.Chem import Recap, Descriptors, rdMolDescriptors, Lipinski, Crippen
from rdkit.Chem.Fraggle import FraggleSim

from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger

uncharger = Uncharger()

def smiles_to_mol(smiles):
    smiles = smiles.replace('*', 'C')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    #standardize charges
    mol = uncharger.uncharge(mol)

    # sanity checks
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 and atom.GetAtomicNum() != 7 and atom.GetAtomicNum() != 8:
            print('WARNING: Molecule contains charged atom that is not N or O')

    # protonate at specific pH of 7.4 using obabel
    protonated_smiles = Chem.MolToSmiles(mol)
    cmd = f'obabel -:"{protonated_smiles}" -ismi -ocan -p{7.4}'
    cmd_return = subprocess.run(cmd, capture_output=True, shell=True)
    protonated_smiles = cmd_return.stdout.decode('utf-8').strip()
    mol = Chem.MolFromSmiles(protonated_smiles)
    assert(all(atom.GetAtomicNum() != 1 for atom in mol.GetAtoms()))

    mol = Chem.AddHs(mol)

    # embedding
    AllChem.EmbedMolecule(mol, randomSeed=42)
    assert(mol.GetNumConformers() != 0)
    # AllChem.EmbedMultipleConfs(mol, randomSeed=42, useRandomCoords=False)
    AllChem.UFFOptimizeMolecule(mol)
    # AllChem.ComputeGasteigerCharges(mol, nIter=12, throwOnParamFailure=True)
    Chem.AssignStereochemistryFrom3D(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True)
    return mol

def validate_molecule(mol):
    return mol is not None and mol.GetNumAtoms() > 2 and mol.GetNumBonds() > 0

def is_multi_molecule(mol):
    return len(Chem.GetMolFrags(mol)) > 1

def is_trivial_fragment(mol):
    heavy = mol.GetNumHeavyAtoms() # number of non-hydrogens
    if heavy < 5:
        return True
    if all(a.GetAtomicNum() == 6 for a in mol.GetAtoms()) and heavy <= 8:
        return True
    return False

def sanitize_dummy_atoms(mol):
    mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Dummy atom like [*]
            atom.SetAtomicNum(6)      # Replace with C
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
    return mol    

def mol_to_smiles(mol, finalize=False):
    if finalize:
        mol = sanitize_dummy_atoms(mol)
        mol = Chem.RemoveHs(mol)
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True) # TODO what is happenign here
    return Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=not finalize, canonical=finalize)

class fragment:
    def __init__(self, molecule, string):
        # print("fragment constructor")
        # print("Making fragment from string ", string)
        self.mol = molecule
        self.num_rotatable_bonds = None
        if molecule:
            # print("mol found with ", Descriptors.NumRadicalElectrons(self.mol))
            # print(Descriptors.MolWt(molecule))
            self.score = 0 # higher score = better fragment
            self.mass = Descriptors.MolWt(molecule)
            self.num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(self.mol)
            self.num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(self.mol)
            # assert Descriptors.NumRadicalElectrons(self.mol) == 0, "too many radicals"
            # print("sanity check OK")
        else:
            self.score = -math.inf
            self.mass = None
        self.docking_score = None
        self.ligand_efficiency = None
        self.log_s = None
        self.smiles = string
        self.synthetic_accessibility = None
        # print("constructor done")

    def __repr__(self):
        return mol_to_smiles(self.mol, True) + f"\n\tcurrent score {self.score}, logs {self.log_s}, docking {self.docking_score}"
    
def get_aromatic_proportion(frag: fragment) -> float:    
    return len(list(frag.mol.GetAromaticAtoms())) / frag.mol.GetNumHeavyAtoms()

def get_recap_frags(molecule):
    recap_tree = Recap.RecapDecompose(molecule)
    fragments = []

    if recap_tree:
        leaves = recap_tree.GetLeaves()
        if leaves:
            for smile, node in leaves.items():
                # Properly handle wildcard atoms
                cleaned_smile = smile.replace('*', 'C')  # Replace wildcard with carbon
                fragments.append(fragment(smiles_to_mol(cleaned_smile)))
        else:
            print("No leaves found in the Recap tree.")
    else:
        print("Failed to obtain Recap decomposition.")
    
    return fragments

def get_brics_frags(molecule):
    assert Descriptors.NumRadicalElectrons(molecule) == 0 
    brics_result = BRICS.BRICSDecompose(molecule, returnMols=True, singlePass=True)
    converted_brics_result = []
    for frag in brics_result:
        assert Descriptors.NumRadicalElectrons(frag) == 0
        cleaned_frag = smiles_to_mol(mol_to_smiles(frag, True))
        converted_brics_result.append(fragment(cleaned_frag))
    return converted_brics_result

def get_fraggle_frags(molecule):
    fraggle_result = FraggleSim.generate_fraggle_fragmentation(molecule)
    converted_fraggle_result = []
    for frag in fraggle_result:
        converted_fraggle_result.append(fragment(smiles_to_mol(frag)))
    return converted_fraggle_result

def get_brics_frags_smiles(molecule):
    assert(Descriptors.NumRadicalElectrons(molecule) == 0)
    brics_result = BRICS.BRICSDecompose(molecule, returnMols=True)
    converted_brics_result = []
    for frag in brics_result:
        assert(Descriptors.NumRadicalElectrons(frag) == 0)
        converted_brics_result.append(mol_to_smiles(frag, True))
    return converted_brics_result

def get_recap_frags_smiles(molecule):
    recap_tree = Recap.RecapDecompose(molecule)
    fragments = []
    
    if recap_tree:
        leaves = recap_tree.GetLeaves()
        if leaves:
            for smile, node in leaves.items():
                # Properly handle wildcard atoms
                cleaned_smile = smile.replace('*', 'C')  # Replace wildcard with carbon
                fragments.append(cleaned_smile)
        else:
            print("No leaves found in the Recap tree.")
    else:
        print("Failed to obtain Recap decomposition.")
    
    return fragments

# def get_medchemfrag_frags_smiles(molecule):
    # the authors of MedChemFrag made their code take SMARTS strings instead of SMILES strings so we have to convert them first
    # drug_smarts = AllChem.MolToSmarts(molecule)
