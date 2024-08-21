import os
import sys
import multiprocessing

def run_task(model, dataset):
    command = (
        f'python controller.py --dataset video --arch {model}_{dataset} '
        f'--batch_size 4 --profile_dir ./profile_pickles_bs '
        f'--bootstrap_pickle_path ../bootstrap_pickles/bootstrap_{dataset}_{model}.pickle '
        f'--data_dir {dataset} --num_classes 2 --model_dir ../models '
        f'--batching_scheme uniform --simulation_pickle_path '
        f'../simulation_pickles/{dataset}_{model}.pickle --batching_scheme clockwork '
        f'--batch_decision_path ../batch_decisions/{model}_1_fixed_30.pickle --slo 1 --qps 30'
    )
    print(f"Running: {dataset} {model}")
    os.system(command)
    os.system(f'tail -n 1 ./logs/output_{model}_{dataset}_video.log >> output_cv.txt')

if __name__ == '__main__':
    models = ['resnet18', 'resnet50', 'resnet101', 'vgg11', 'vgg13', 'vgg16']
    datasets = ['auburn', 'hampton', 'oxford', 'calgary', 'coral', 'ohio', 'bellevue1', 'bellevue2']

    os.system('rm output_cv.txt')
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 4)

    try:
        pool.starmap(run_task, [(model, dataset) for dataset in datasets for model in models])
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pool.close()
        pool.join()