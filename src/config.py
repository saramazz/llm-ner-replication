# Read the current file path

import os


# define global path
current_path = os.path.dirname(os.path.realpath(__file__))

global_path = os.path.abspath(os.path.join(current_path, os.pardir))

saving_path = os.path.join(global_path, "saved_results")

corpora_path = os.path.join(global_path, "Extr_Val_R/Corpora_python_ready")
translated_corpora_path = os.path.join(global_path, "Extr_Val_R/Corpora_translated")

"""
# print all these paths

print(f"Current path: {current_path}")
print(f"Global path: {global_path}")
print(f"Saving path: {saving_path}")
print(f"Corpora path: {corpora_path}")
"""
