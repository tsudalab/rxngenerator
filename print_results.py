import sys
import gzip
import pickle
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
#import sascorer
import sys
from cairosvg import svg2png



filenames =["qed1.txt","qed2.txt","qed3.txt","qed4.txt","qed5.txt","qed6.txt","qed7.txt","qed8.txt","qed9.txt","qed10.txt"]
all_smiles =[]
all_scores =[]
all_reactions=[]
for filename in filenames:
	with open("Results/"+filename) as reader:
		lines = reader.readlines()
		for line in lines:
			line = line.strip()
			res = line.split(" ")
			smiles = res[0]
			score = float(res[-1])
			reactions = res[1:-1]
			all_smiles.append(smiles)
			all_scores.append(score)
			all_reactions.append(reactions)

			#print(smiles, score)


# filter
b_all_smiles =[]
b_all_scores =[]
b_all_reactions=[]
for i in range(len(all_smiles)):
	if all_smiles[i] not in b_all_smiles:
		b_all_smiles.append(all_smiles[i])
		b_all_reactions.append(all_reactions[i])
		b_all_scores.append(all_scores[i])
pairs = [(smiles, rxn,-score) for smiles, rxn, score in zip(b_all_smiles, b_all_reactions, b_all_scores)]
pairs = sorted(pairs, key=lambda x:x[2], reverse=True)

mols = [Chem.MolFromSmiles(s) for s, _, _ in pairs[:50]]
vals = ["%.4f" % score for _,_,score in pairs[:50]]
#print(mols)
img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200,135), legends=vals, useSVG=True)
svg2png(bytestring=img,write_to='output.png')
#print(mols)
for smiles, reactions, score in pairs[:50]:
	print(smiles, reactions, score)