import pickle


filename = "../profile_pickles_bs/bert-base-uncased_16_earlyexit_profile.pickle"

with open(filename, "rb") as f:
    p = pickle.load(f)
    
print(p)

profile_sum = p.average_transformer_layers(p.children[0].children[1].children[0].children)

print(profile_sum[0])

