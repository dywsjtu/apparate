import torch
import numpy as np


def load_weights(backbone_model_dir):
    loaded_state_dict = torch.load(f"{backbone_model_dir}/pytorch_model.bin")  # , map_location="cpu"
    loaded_keys = list(loaded_state_dict.keys())
    print(f"loaded_keys: {loaded_keys}")
    return loaded_state_dict

    # for expected_key in expected_keys:
    #     if "branched_module" in expected_key:
    #         expected_key_wo_bp = expected_key.replace("branched_module.", "")
    #         assert loaded_state_dict[expected_key_wo_bp].shape == model_state_dict[expected_key].shape, \
    #             f"Mismatch between tensor sizes: {expected_key_wo_bp} ({loaded_state_dict[expected_key_wo_bp].shape}) != {expected_key} ({model_state_dict[expected_key].shape})!"
    #         print(f"copying tensor {loaded_state_dict[expected_key_wo_bp].shape} from {expected_key_wo_bp} to {expected_key}")
    #         model.state_dict()[expected_key].data.copy_(loaded_state_dict[expected_key_wo_bp])
    # return model


backbone_model_dir1 = "../../deebert/saved_models_EE/bert-base/RTE/two_stage"  # EE model
# backbone_model_dir2 = "../../deebert/saved_models_vanilla/bert-base/RTE/two_stage"  # vanilla model
backbone_model_dir2 = "../../model_checkpoints/huggingface/bert-base/RTE/two_stage"  # vanilla model

weights1 = load_weights(backbone_model_dir1)
weights2 = load_weights(backbone_model_dir2)


for expected_key in weights1.keys():
    expected_key_wo_bp = expected_key.replace("branched_module.", "")
    # print(expected_key, expected_key_wo_bp)
    if expected_key_wo_bp in weights2:
        assert weights2[expected_key_wo_bp].shape == weights1[expected_key].shape
        # print(type(weights2[expected_key_wo_bp]))
        # print(type(weights2[expected_key_wo_bp]))
        # if expected_key_wo_bp == "bert.encoder.layer.11.attention.self.key.bias":
        if not torch.equal(weights2[expected_key_wo_bp], weights1[expected_key]):
            print(f"expected_key {expected_key}, shape {weights2[expected_key_wo_bp].shape}")
            print(weights2[expected_key_wo_bp])
            print("="*50)
            print(weights1[expected_key])
            print("="*50)
            eq_arr = torch.eq(weights2[expected_key_wo_bp], weights1[expected_key])
            eq_arr = eq_arr.cpu().detach().numpy()
            count = np.count_nonzero(eq_arr)
            print(eq_arr)
            print(count)
            exit(0)
    else:
        assert "branch_net" in expected_key
