!zip -r outputs.zip outputs data/raw/metadata.csv
from google.colab import files
files.download("outputs.zip")
