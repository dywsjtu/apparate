import os
import sys
import pickle
sys.path.insert(1, os.path.join(os.path.dirname(os.getcwd()), 'profiling'))  # for loading profile pickles

"""
Updates old versions of profile pickles so that they are compatible with the latest codebase
"""

# profile_pickle_dir = "../profile_pickles_bs"
profile_pickle_dir = "../yinwei_profile_pickles_bs"
listdir = [os.path.join(profile_pickle_dir, f) for f in os.listdir(profile_pickle_dir)]
filenames = list(filter(lambda f: os.path.isfile(f) and f.endswith(".pickle"), listdir))

def update_profile(p):
    """Recursively modifies certain attributes of a profile object.
    Currently, we modify:
        self.vanilla_latency_after_me -> self.vanilla_latency_up_until_me
        self.ramp_latencies_after_me -> self.ramp_latencies_up_until_me

    Args:
        p (Profile): Profile object.
    
    Returns:
        updated_me (bool): whether any updates were made
    """
    # print(dir(p))
    updated_me = False
    
    old_new_attr_names = [
        ("vanilla_latency_after_me", "vanilla_latency_up_until_me"),
        ("ramp_latencies_after_me", "ramp_latencies_up_until_me"),
    ]
    
    for old_name, new_name in old_new_attr_names:
        if hasattr(p, old_name):
            assert not hasattr(p, new_name)
            setattr(p, new_name, getattr(p, old_name))
            delattr(p, old_name)
            updated_me = True

    for child in p.children:
        updated_child = update_profile(child)
        updated_me = updated_me or updated_child
    
    return updated_me


for filename in filenames:
    print('='*50)
    print(f"updating file {filename}...")
    with open(filename, "rb") as f:
        p = pickle.load(f)
    
    updated = update_profile(p)
    if not updated:
        print(f"warning: file {filename} received no updates")
    # print(p)
    with open(filename, "wb") as f:
        pickle.dump(p, f)
    print_to_filename = filename[:-len("pickle")] + "txt"
    if not os.path.exists(print_to_filename):
        print(p, file=open(print_to_filename, "w"))
