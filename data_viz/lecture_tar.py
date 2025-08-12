import tarfile

# Chemins
source = "E:/MindBridge.tar"
destination = r"\\?\E:/Mb"  # Crée un dossier à la racine, pour éviter les chemins trop longs

# Extraction
with tarfile.open(source, "r") as tar:
    tar.extractall(path=destination)
    print("Extraction réussie dans :", destination)
