import torchvision.models as models
import torch
from torchinfo import summary
# from deepspeed.profiling.flops_profiler import get_model_profile

from profiler import FlopsProfiler

with torch.cuda.device(0):
    model = models.resnet18()
    batch_size = 16
    input_size = (batch_size, 3, 224, 224)
    # run torchinfo first to fact-check the input/output shape
    # summary(model, input_size=input_size)  # running this first will create a bug that blocks deepspeed profiler from running

    ################################################################
    # # profile using deepspeed's flops profiler
    # flops, macs, params = get_model_profile(model=model, # model
    #                                 input_shape=(batch_size, 3, 224, 224), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
    #                                 args=None, # list of positional arguments to the model.
    #                                 kwargs=None, # dictionary of keyword arguments to the model.
    #                                 print_profile=True, # prints the model graph with the measured profile attached to each module
    #                                 detailed=True, # print the detailed profile
    #                                 module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
    #                                 top_modules=1, # the number of top modules to print aggregated profile
    #                                 warm_up=10, # the number of warm-ups before measuring the time of each module
    #                                 as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
    #                                 output_file=None, # path to the output file. If None, the profiler prints to stdout.
    #                                 ignore_modules=None) # the list of modules to ignore in the profiling
    ################################################################
    # # profile using our own modified version of the deepspeed profiler
    inputs = torch.randn(input_size)
    prof = FlopsProfiler(model)
    prof.start_profile()
    model(inputs)
    profile = prof.generate_profile()
    print(profile)
    prof.print_model_profile()
    prof.end_profile()
    ################################################################
