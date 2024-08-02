from PIL import Image
import glob

for file in glob.glob("*.webp"):
    image = Image.open(file).convert("RGB")
    image.save(file.replace("webp", "jpg"))