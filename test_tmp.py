import os
from pathlib import Path

print("Checking /tmp")
for subdir in Path("/tmp").iterdir():
    if subdir.is_dir():
        print(f"DIR: {subdir.name}")
