import yaml
import timeit
import argparse
import numpy as np
import pandas as pd
from utils.modelnames import models
from experiment.execute import execute
from utils.argcheck import check_int_positive
from utils.io import load_numpy, save_dataframe_csv, find_best_hyperparameters, load_yaml


def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']

    R_train = load_numpy(path=args.path, name=args.document + args.train)
    R_valid = load_numpy(path=args.path, name=args.document + args.valid)
    R_rtrain = load_numpy(path=args.path, name=args.document + args.unif_train)
    R_test = load_numpy(path=args.path, name=args.document + args.test)

    if args.searcher == 'grid':
        df = find_best_hyperparameters(table_path+args.problem, args.main_metric)

        frame = []
        for idx, row in df.iterrows():
            start = timeit.default_timer()
            row = row.to_dict()
            row['metric'] = ['AUC']
            result = execute(R_train, R_valid, R_rtrain, R_test, row, models[row['model']], source=row['source'],
                             seed=args.seed, folder=args.model_folder+args.document)
            stop = timeit.default_timer()
            print('Time: ', stop - start)
            frame.append(result)

    else:
        parameters_dict = yaml.safe_load(
            open(table_path+args.problem+'op_hyper_params.yml', 'r'))['PCIC21']
        model_list = parameters_dict.keys()

        frame = []
        for model in model_list:
            init_row = {'corruption': 0, 'alpha': 1.0, 'model': model, 'rank': 100,
                        'gamma': 1.0, 'iter': 500, 'batch_size': 1024, 'lambda': 0.0001, 'clip': 1.0,
                        'learning_rate': 0.001, 'source': None}
            row = init_row.copy()
            row.update(parameters_dict[model])
            row['rank'] = np.int(row['rank'])
            row['batch_size'] = np.int(row['batch_size'])

            start = timeit.default_timer()
            row['metric'] = ['AUC']
            result = execute(R_train, R_valid, R_rtrain, R_test, row, models[row['model']], source=row['source'],
                             seed=args.seed, folder=args.model_folder + args.document)
            stop = timeit.default_timer()
            print('Time: ', stop - start)
            frame.append(result)

    results = pd.concat(frame)
    save_dataframe_csv(results, table_path, args.problem+args.table_name)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Reproduce")
    parser.add_argument('-tb', dest='table_name', default="final_result.csv")
    parser.add_argument('-otb', dest='op_table_name', default="op_final_result.csv")
    parser.add_argument('-p', dest='path', default="datax/")
    parser.add_argument('-d', dest='document', default="processed/")
    parser.add_argument('-t', dest='train', default='user_big_mat.npz')
    parser.add_argument('-v', dest='valid', default='valid.npz')
    parser.add_argument('-ut', dest='unif_train', default='user_choice_mat.npz')
    parser.add_argument('-e', dest='test', default='test.npz')
    parser.add_argument('-mf', dest='model_folder', default='latent/')  # Model saving folder
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=2021)
    parser.add_argument('-searcher', dest='searcher', default="optuna")
    args = parser.parse_args()

    main(args)
