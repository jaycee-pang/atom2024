from rdkit import Chem
from rdkit.Chem import MolStandardize


def normalize(mol) -> Chem.rdchem.Mol:
    return MolStandardize.rdMolStandardize.CanonicalTautomer(
        MolStandardize.rdMolStandardize.ChargeParent(mol)
    )


def dedup_molecules(mols) -> list[Chem.rdchem.Mol]:

    inchi_dict = {}
    rmols = []

    for mol in mols:
        inchi_key = Chem.MolToInchiKey(mol)
        if (not inchi_dict.get(inchi_key)):
            inchi_dict[inchi_key] = mol

    for mol in inchi_dict.values():
        rmols.append(mol)

    return rmols
