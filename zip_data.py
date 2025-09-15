import json, os
import shutil

galleryPath = "coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery"

os.rename(os.path.join(galleryPath, "gallery.json"), os.path.join(galleryPath, "gallery_mothra.json"))
os.remove("coupled_flux_qubit_protocol/myarch_mothra.zip")

# zip file
# zip_name = "coupled_flux_qubit_protocol/myarch"
# shutil.make_archive(zip_name, 'zip', galleryPath)
os.system(f"zip -rj coupled_flux_qubit_protocol/myarch_mothra.zip {galleryPath}")

# clean up
os.system("rm -rf ~/source/coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/*")
os.system("touch ~/source/coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/gallery.json")
os.system("echo [] >> ~/source/coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/gallery.json")
