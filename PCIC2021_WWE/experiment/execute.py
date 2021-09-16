import os
import math
import numpy as np
import pandas as pd
from models.predictor import predict
from scipy.sparse import lil_matrix
from utils.progress import WorkSplitter
from sklearn.metrics import roc_auc_score


def execute(train, validation, rtrain, test, params, model, source=None, seed=0,folder='latent'):
    progress = WorkSplitter()

    columns = ['model', 'rank', 'alpha', 'lambda', 'clip', 'iter', 'corruption', 'gamma', 'batch_size', 'learning_rate',
               'source']

    progress.section("\n".join([":".join((str(k), str(params[k]))) for k in columns]))

    df = pd.DataFrame(columns=columns)

    if not os.path.exists(folder):
        os.makedirs(folder)

    if isinstance(source, str):
        if os.path.isfile('{2}/{3}_U_{0}_{1}.npy'.format(params['model'], params['rank'], folder, source)):

            U = np.load('{2}/{3}_U_{0}_{1}.npy'.format(params['model'], params['rank'], folder, source))
            V = np.load('{2}/{3}_V_{0}_{1}.npy'.format(params['model'], params['rank'], folder, source))

            if os.path.isfile(
                    '{2}/{3}_uB_{0}_{1}.npy'.format(params['model'], params['rank'], folder, source)):
                uB = np.load(
                    '{2}/{3}_uB_{0}_{1}.npy'.format(params['model'], params['rank'], folder, source))
            else:
                uB = None

            if os.path.isfile(
                    '{2}/{3}_vB_{0}_{1}.npy'.format(params['model'], params['rank'], folder, source)):
                vB = np.load(
                    '{2}/{3}_vB_{0}_{1}.npy'.format(params['model'], params['rank'], folder, source))
            else:
                vB = None

        else:
            U, Vt, uB, vB = model(train,
                                  validation,
                                  matrix_utrain=rtrain,
                                  iteration=params['iter'],
                                  rank=params['rank'],
                                  batch_size=params['batch_size'],
                                  learning_rate=params['learning_rate'],
                                  lam=params['lambda'],
                                  clip=params['clip'],
                                  alpha=params['alpha'],
                                  corruption=params['corruption'],
                                  gamma=params['gamma'],
                                  seed=seed,
                                  source=source)
            V = Vt.T
            np.save('{2}/{3}_U_{0}_{1}'.format(params['model'], params['rank'], folder, source), U)
            np.save('{2}/{3}_V_{0}_{1}'.format(params['model'], params['rank'], folder, source), V)
            if uB is not None:
                np.save('{2}/{3}_uB_{0}_{1}'.format(params['model'], params['rank'], folder, source), uB)
            if vB is not None:
                np.save('{2}/{3}_vB_{0}_{1}'.format(params['model'], params['rank'], folder, source), vB)

    else:
        if os.path.isfile('{2}/U_{0}_{1}.npy'.format(params['model'], params['rank'], folder)):

            U = np.load('{2}/U_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
            V = np.load('{2}/V_{0}_{1}.npy'.format(params['model'], params['rank'], folder))

            if os.path.isfile('{2}/uB_{0}_{1}.npy'.format(params['model'], params['rank'], folder)):
                uB = np.load('{2}/uB_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
            else:
                uB = None

            if os.path.isfile('{2}/vB_{0}_{1}.npy'.format(params['model'], params['rank'], folder)):
                vB = np.load('{2}/vB_{0}_{1}.npy'.format(params['model'], params['rank'], folder))
            else:
                vB = None

        else:
            U, Vt, uB, vB = model(train,
                                  validation,
                                  matrix_utrain=rtrain,
                                  iteration=params['iter'],
                                  rank=params['rank'],
                                  batch_size=params['batch_size'],
                                  learning_rate=params['learning_rate'],
                                  lam=params['lambda'],
                                  clip=params['clip'],
                                  alpha=params['alpha'],
                                  corruption=params['corruption'],
                                  gamma=params['gamma'],
                                  seed=seed)
            V = Vt.T

            np.save('{2}/U_{0}_{1}'.format(params['model'], params['rank'], folder), U)
            np.save('{2}/V_{0}_{1}'.format(params['model'], params['rank'], folder), V)
            if uB is not None:
                np.save('{2}/uB_{0}_{1}'.format(params['model'], params['rank'], folder), uB)
            if vB is not None:
                np.save('{2}/vB_{0}_{1}'.format(params['model'], params['rank'], folder), vB)

    progress.subsection("Prediction")

    rating_prediction = predict(matrix_U=U, matrix_V=V,  bias_U=uB, bias_V=vB, matrix_Test=test)

    progress.subsection("Evaluation")
    test_ui_pairs = lil_matrix(test)
    test_ui_pairs = np.asarray(test_ui_pairs.nonzero()).T.astype('int32')
    test_label = np.asarray(test[test_ui_pairs[:, 0], test_ui_pairs[:, 1]])[0]
    test_label[test_label == -1] = 0

    result = roc_auc_score(test_label, rating_prediction)

    result_dict = params

    result_dict[params['metric']] = [round(result, 4)]
    df = df.append(result_dict, ignore_index=True)

    return df
