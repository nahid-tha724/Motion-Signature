import shutil
import os

# Define the directory containing the figures and the output zip name
figures_dir = "outputs/figures"
zip_filename = "outputs/figures_all"

# Create a zip archive of the figures directory
shutil.make_archive(zip_filename, 'zip', figures_dir)
print(f"Zip archive created at: {zip_filename}.zip")

# Optional: If you are running this inside Google Colab,
# this line will automatically prompt your browser to download the file to your PC:
from google.colab import files
files.download(f"{zip_filename}.zip")
