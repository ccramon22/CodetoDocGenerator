# Python file to fix the dependency issue
import os
import sys

# Create requirements.txt
requirements = """
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
transformers==4.31.0
datasets==2.14.4
pandas
numpy
gitpython
peft==0.5.0
bitsandbytes==0.41.1
accelerate==0.21.0
intel-extension-for-pytorch==2.0.100 # For Intel GPUs
"""

# Write requirements file
with open("requirements.txt", "w") as f:
    f.write(requirements)

print("Created requirements.txt with compatible versions")
print("Next, run: pip install -r requirements.txt --force-reinstall")