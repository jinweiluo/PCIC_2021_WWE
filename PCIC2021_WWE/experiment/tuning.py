import os
import time
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from models.predictor import predict
from utils.progress import WorkSplitter
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml
from sklearn.metrics import roc_auc_score


def hyper_parameter_tuning(train, validation, test, rtrain, rating, item2tag, pair_samples, params, save_path,
                           op_save_path, source=None, seed=0, searcher='grid'):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']
    if not os.path.exists(table_path):
        os.makedirs(table_path)

    if searcher == 'grid':
        if not os.path.exists(table_path+save_path):
            df = pd.DataFrame(columns=['models', 'rank', 'alpha', 'lambda', 'clip', 'iter', 'corruption', 'gamma',
                                       'batch_size', 'learning_rate', 'source'])
        else:
            df = load_dataframe_csv(table_path, save_path)

        for algorithm in params['models']:

            for rank in params['rank']:

                for alpha in params['alpha']:

                    for lam in params['lambda']:

                        for clip in params['clip']:

                            for corruption in params['corruption']:

                                for gamma in params['gamma']:

                                    for batch_size in params['batch_size']:

                                        for learning_rate in params['learning_rate']:

                                            if ((df['models'] == algorithm) &
                                                (df['rank'] == rank) &
                                                (df['alpha'] == alpha) &
                                                (df['lambda'] == lam) &
                                                (df['clip'] == clip) &
                                                (df['corruption'] == corruption) &
                                                (df['gamma'] == gamma) &
                                                (df['batch_size'] == batch_size) &
                                               (df['learning_rate'] == learning_rate)).any():
                                                continue

                                            format = "models: {0}, rank: {1}, alpha: {2}, lambda: {3}, " + \
                                                     "clip: {4}, corruption: {5}, gamma: {6}, " + \
                                                     "batch_size: {7}, learning_rate: {8}"
                                            progress.section(format.format(algorithm, rank, alpha, lam, clip, corruption,
                                                                           gamma, batch_size, learning_rate))
                                            U, Vt, uB, vB = params['models'][algorithm](train,
                                                                                        validation,
                                                                                        matrix_test=test,
                                                                                        matrix_rating=rating,
                                                                                        item2tag=item2tag,
                                                                                        matrix_utrain=rtrain,
                                                                                        pair_samples=pair_samples,
                                                                                        iteration=params['iter'],
                                                                                        rank=rank,
                                                                                        batch_size=batch_size,
                                                                                        learning_rate=learning_rate,
                                                                                        lam=lam,
                                                                                        clip=clip,
                                                                                        alpha=alpha,
                                                                                        corruption=corruption,
                                                                                        gamma=gamma,
                                                                                        seed=seed,
                                                                                        source=source)
                                            V = Vt.T

                                            progress.subsection("Prediction")

                                            rating_prediction, valid_label = predict(matrix_U=U, matrix_V=V, bias_U=uB,
                                                                                     bias_V=vB, matrix_Test=validation,
                                                                                     gpu=True)

                                            progress.subsection("Evaluation")
                                            result = roc_auc_score(valid_label, rating_prediction)

                                            result_dict = {'models': algorithm, 'rank': rank, 'alpha': alpha, 'lambda': lam,
                                                           'clip': clip, 'iter': params['iter'],
                                                           'corruption': corruption, 'gamma': gamma,
                                                           'batch_size': batch_size, 'learning_rate': learning_rate,
                                                           'source': source, 'AUC': round(result, 4)}

                                            df = df.append(result_dict, ignore_index=True)

                                            save_dataframe_csv(df, table_path, save_path)

    if searcher == 'optuna':

        for algorithm in params['models']:
            trials, best_params = params['models'][algorithm](train,
                                                              validation,
                                                              matrix_test=test,
                                                              matrix_rating=rating,
                                                              item2tag=item2tag,
                                                              matrix_utrain=rtrain,
                                                              pair_samples=pair_samples,
                                                              iteration=params['iter'],
                                                              seed=seed,
                                                              source=source,
                                                              searcher='optuna')

        trials.to_csv(table_path+op_save_path)

        if Path(table_path + 'op_hyper_params.yml').exists():
            pass
        else:
            yaml.dump(dict(PCIC21=dict()),
                      open(table_path + 'op_hyper_params.yml', 'w'), default_flow_style=False)
        time.sleep(0.5)
        hyper_params_dict = yaml.safe_load(open(table_path + 'op_hyper_params.yml', 'r'))
        hyper_params_dict['PCIC21'][algorithm] = best_params
        yaml.dump(hyper_params_dict, open(table_path + 'op_hyper_params.yml', 'w'),
                  default_flow_style=False)