import os
import sys
import multiprocessing

def run_task(dataset, model, batch_decision_path):
    command = (
        f'python controller.py '
        f'--dataset {dataset} --arch {model} --batch_size 1 '
        f'--profile_dir ./profile_pickles_bs '
        f'--num_classes 2 --model_dir ../ --batching_scheme clockwork '
        f'--simulation_pickle_path ../simulation_pickles/{dataset}_{model}.pickle '
        f'--bootstrap_pickle_path ../bootstrap_pickles/bootstrap_{dataset}_{model}.pickle '
        f'--batch_decision_path {batch_decision_path} --slo 2 --qps 30'
    )
    print(f"Running: {dataset} {model}")
    os.system(command)
    os.system(f'tail -n 2 ./logs/output_{model}_{dataset}.log >> output_nlp.txt')

if __name__ == '__main__':
    models = ['distilbert-base', 'bert-base', 'bert-large', 'gpt2-medium']
    # models = ['gpt2-medium']
    datasets = ['amazon_reviews', 'imdb']

    batch_decision_dict = {
        'distilbert-base': "../batch_decisions/distilbert-base_azure.pickle",
        'bert-base': "../batch_decisions/bert-base_azure.pickle",
        'bert-large': "../batch_decisions/bert-large_azure.pickle",
        'gpt2-medium': "../batch_decisions/gpt2-medium_azure.pickle",
    }
    
    os.system('rm output_nlp.txt')

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 4)

    try:
        tasks = [
            (dataset, model, batch_decision_dict[model])
            for dataset in datasets
            for model in models
        ]
        pool.starmap(run_task, tasks)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pool.close()
        pool.join()