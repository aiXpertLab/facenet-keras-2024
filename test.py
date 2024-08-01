import sys
import os

print("Current Working Directory:", os.getcwd())
print("sys.path:", sys.path)

try:
    import utils
    print("Successfully imported utils")
except Exception as e:
    print("Failed to import utils:", e)

print('utils' in sys.modules)