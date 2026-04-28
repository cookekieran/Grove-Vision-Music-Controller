"""
This script organises raw training data from edgeimpulse.com, labelling images with their 
respective classes. Images for each label are grouped into a folder.
"""

import os
import shutil

source_folder = './data/training'

def organize_data():
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    count = 0
    for filename in files:
        if filename.endswith('.py'):
            continue
            
        parts = filename.split('.')
        category = parts[0]
        category_path = os.path.join(source_folder, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
            
        src = os.path.join(source_folder, filename)
        dst = os.path.join(category_path, filename)
        
        try:
            shutil.move(src, dst)
            count += 1
        except Exception as e:
            print(f"Error moving {filename}: {e}")

    print(f"Organised {count} files into their respective folders.")

if __name__ == "__main__":
    organize_data()