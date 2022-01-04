import pymongo
client = pymongo.MongoClient("localhost", 27017)



def collection_example_reactions_smilesonly():
	db = client['reaction_examples']
	collection = db['lowe_1976-2013_USPTOgrants']
	return collection

def collection_example_reactions_details():
    db = client['reaction_examples']
    collection = db['lowe_1976-2013_USPTOgrants_reactions']
    return collection

def collection_templates():
    db = client['askcos_transforms']
    collection = db['lowe_refs_general_v3']
    return collection

def collection_candidates():
    db = client['prediction']
    collection = db['candidate_edits_8_9_16']
    return collection 



#col1 = collection_example_reactions_smilesonly()
#col2 = collection_example_reactions_details()
#col3 = collection_templates()
#col4 = collection_candidates()
#print(col4)