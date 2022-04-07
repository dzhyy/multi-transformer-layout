import os
import shutil

def remove_file(file_path):
    if os.path.exists(file_path) is True:
        os.remove(file_path)

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        create_folder(folder_path)

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)