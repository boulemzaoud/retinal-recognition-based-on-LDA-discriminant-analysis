
"""
I/O utilities – loading datasets, saving reports, exporting images, etc.
Group your filesystem helpers here to keep the rest of the code clean.
"""
import os
import csv
import pickle

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
