import argparse
from rdkit import Chem

parser = argparse.ArgumentParser()
parser.add_argument('input_filename')
parser.add_argument('output_filename')

args = parser.parse_args()

suppl = Chem.SDMolSupplier(args.input_filename)
w = Chem.SDWriter(args.output_filename)

inchi_dict = {}

for mol in suppl:
    inchi_key = Chem.MolToInchiKey(mol)
    if (not inchi_dict.get(inchi_key)):
        inchi_dict[inchi_key] = mol

for mol in inchi_dict.values():
    w.write(mol)
