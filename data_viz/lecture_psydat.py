import pickle

with open(r"G:\experiment\datalogs\007_test_building_pipeline_2023-08-24_17h32.41.107.psydat", "rb") as f:
    data = pickle.load(f)

# Supprimer l'attribut si présent
if hasattr(data, 'connectedSaveMethods'):
    delattr(data, 'connectedSaveMethods')
if hasattr(data, 'thisExp'):  # Parfois, un sous-objet peut aussi causer le problème
    if hasattr(data.thisExp, 'connectedSaveMethods'):
        delattr(data.thisExp, 'connectedSaveMethods')

print(data)