import os
import shutil
from google.colab import files

# Ensure directories exist so shutil doesn't throw an error if data/processed is missing
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# Create the zip archive cleanly using Python's shutil
output_filename = "final_project_outputs"
archive_format = "zip"

# Archive the 'outputs' directory
shutil.make_archive(output_filename, archive_format, root_dir=".", base_dir="outputs")

print(f"Archive created successfully: {output_filename}.{archive_format}")

# Trigger the download in Colab
files.download(f"{output_filename}.{archive_format}")
