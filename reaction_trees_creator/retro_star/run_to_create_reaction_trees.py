import numpy as np
import torch
import random
import logging
import time
import pickle
import os
from retro_star.common import args, prepare_starting_molecules, prepare_mlp, \
    prepare_molstar_planner, smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.api import RSPlanner


import rdkit.Chem as Chem 
from rdkit.Chem import AllChem
import pickle
import pandas as pd

from database import collection_templates, collection_example_reactions_smilesonly
TRANSFORM_DB = collection_templates()
REACTION_DB = collection_example_reactions_smilesonly()
reaction_smiles_field = 'reaction_smiles'


if __name__ == '__main__':
    n_reactions = REACTION_DB.count()
    N = 1000000
    i = -1
    N = min([N, n_reactions])
    target_mols = []

    for example_doc in REACTION_DB.find(no_cursor_timeout = True):
        i+=1
        if i == N:
            i-=1
            break
        reaction_smiles = str(example_doc[reaction_smiles_field])
        reactants, agents, products = [x for x in [mols.split('.') for mols in reaction_smiles.split('>')]]
        if len(products)==1:
            mol = Chem.MolFromSmiles(products[0])
            if mol == None:
                continue
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.ClearProp('molAtomMapNumber')
            product = Chem.MolToSmiles(mol)
            if product not in target_mols:
                target_mols.append(product)
    


    planner = RSPlanner(
    gpu=-1,
    use_value_fn=True,
    iterations=200,
    expansion_topk=50
    )

    file = open("synthetic_routes.txt", "w")

    for i,target_mol in enumerate(target_mols):
        try:
            result = planner.plan(target_mol)
            if len(result) != 0:
                file.write(" ".join(result) + "\n")
                print(i, target_mol, len(result))
        except:
            print("Error occurred")
    file.close()

