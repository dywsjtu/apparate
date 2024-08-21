# convert nlp pickles in original nlp format to cv format
import os
import sys
import pickle

sys.path.insert(1, os.path.join(os.path.dirname(os.getcwd())))
import utils

orig_dir = "/home/ruipan/deebert/entropy_pickles_hf_oldformat"
new_dir = "/home/ruipan/deebert/entropy_pickles_hf"

for filename in os.listdir(orig_dir):
    orig_filename = os.path.join(orig_dir, filename)
    new_filename = os.path.join(new_dir, filename)
    print(f"converting {filename}")
    with open(orig_filename, "rb") as f:
        p = pickle.load(f)
        
    p = utils.pickle_format_convert(p)
    
    with open(new_filename, "wb") as f:
        pickle.dump(p, f)
