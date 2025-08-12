import pydicom
import matplotlib.pyplot as plt

# Charger le fichier DICOM
fichier = r"C:\Users\Rémi\AppData\Local\Temp\1bddcd2d-3daf-48d8-b168-77026ae2a8af_CARSQR_03-20230814T0845.zip.8af\CARSQR_03-20230814T0845\scans\11-AP_MB4_BOLD_Risk1\resources\DICOM\files\1.3.12.2.1107.5.2.43.66023.30000023081402233684300003860-11-3-1q7y1ig.dcm"
ds = pydicom.dcmread(fichier)

# Afficher les métadonnées de base
print(ds)

# Afficher l'image si présente
if 'PixelData' in ds:
    plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
    plt.title("Image DICOM")
    plt.axis("off")
    plt.show()
else:
    print("Aucune image dans ce fichier.")
