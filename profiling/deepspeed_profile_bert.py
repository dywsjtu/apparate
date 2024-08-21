from functools import partial
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from deepspeed.profiling.flops_profiler import get_model_profile

from profiler import FlopsProfiler


def bert_input_constructor(batch_size, seq_len, tokenizer):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * batch_size,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * batch_size)
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    # inputs: dict with keys "input_ids", "token_type_ids", "attention_mask", "labels"
    return inputs


with torch.cuda.device(0):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    batch_size = 1
    seq_len = 128
    enable_profile = True
    if enable_profile:
      inputs = bert_input_constructor(batch_size, seq_len, tokenizer)
      ################################################################
      # # profile using deepspeed's flops profiler
      # flops, macs, params = get_model_profile(
      #     model,
      #     kwargs=inputs,
      #     print_profile=True,
      #     detailed=True,
      #     module_depth=-1,
      #     warm_up=10
      # )
      # print(f"flops {flops}")
      # print(f"macs {macs}")
      # print(f"params {params}")
      ################################################################
      # # profile using our own modified version of the deepspeed profiler
      prof = FlopsProfiler(model)
      prof.start_profile()
      model(**inputs)

      profile = prof.generate_profile()
      # compare with deepspeed profiler's text output
      print(profile)
      prof.print_model_profile()
      prof.end_profile()
      ################################################################
    else:
      inputs = bert_input_constructor((batch_size, seq_len), tokenizer)
      outputs = model(inputs)

