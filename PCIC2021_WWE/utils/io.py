import yaml
import numpy as np
import pandas as pd
from os import listdir
from ast import literal_eval
from os.path import isfile, join
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz


def load_bigtag(path, name, sep, df_name, item2tag, shape=None):
    df = pd.read_csv(path + name, sep=sep, header=None, names=df_name)
    users, items, tags = df[df_name[0]], df[df_name[1]], df[df_name[2]]
    rows, cols, values = [], [], []

    for i in range(users.shape[0]):
        if tags[i] == -1:
            temp = item2tag[str(items[i])]
            cols.extend(temp)
            rows.extend([users[i]] * len(temp))
            values.extend([-1] * len(temp))
        else:
            cols.append(tags[i])
            rows.append(users[i])
            values.append(1)

    df = pd.DataFrame([rows, cols, values]).transpose()
    df.columns = ['uid', 'tag', 'label']
    df = df.sort_values('label').drop_duplicates(subset=['uid', 'tag'], keep='last')
    mat = df.to_numpy()
    rows, cols, values = mat[:, 0], mat[:, 1], mat[:, 2]
    if shape:
        return csr_matrix((values, (rows, cols)), shape=shape)
    else:
        return csr_matrix((values, (rows, cols)), shape=(np.max(rows) + 1, np.max(cols) + 1))


def load_choicetag(path, name, sep, df_name, item2tag, shape=None, filter_matrix=None):
    df = pd.read_csv(path + name, sep=sep, header=None, names=df_name)
    users, items, tags = df[df_name[0]], df[df_name[1]], df[df_name[2]]
    rows, cols, values = [], [], []

    for i in range(users.shape[0]):
        if tags[i] == -1:
            temp = item2tag[str(items[i])]
            cols.extend(temp)
            rows.extend([users[i]] * len(temp))
            values.extend([-1] * len(temp))
        else:
            cols.append(tags[i])
            rows.append(users[i])
            values.append(1)

            # Unlike bigtag set, tags that do not appear in choicetag set can be considered negative
            remaining_tags = list(set(item2tag[str(items[i])]) - {tags[i]})

            if filter_matrix is None:
                cols.extend(remaining_tags)
                rows.extend([users[i]] * len(remaining_tags))
                values.extend([-1] * len(remaining_tags))
            else:
                for t in remaining_tags:
                    if filter_matrix[users[i], t] == 0:
                        cols.append(t)
                        rows.append(users[i])
                        values.append(-1)

    df = pd.DataFrame([rows, cols, values]).transpose()
    df.columns = ['uid', 'tag', 'label']
    df = df.sort_values('label').drop_duplicates(subset=['uid', 'tag'], keep='last')
    mat = df.to_numpy()
    rows, cols, values = mat[:, 0], mat[:, 1], mat[:, 2]
    if shape:
        return csr_matrix((values, (rows, cols)), shape=shape)
    else:
        return csr_matrix((values, (rows, cols)), shape=(np.max(rows) + 1, np.max(cols) + 1))


def load_valid(path, name, sep, df_name, shape=None):
    df = pd.read_csv(path + name, sep=sep, header=None, names=df_name)
    df.loc[df[df_name[2]] == 0, df_name[2]] = -1
    rows, cols, values = df[df_name[0]], df[df_name[1]], df[df_name[2]]

    if shape:
        return csr_matrix((values, (rows, cols)), shape=shape)
    else:
        return csr_matrix((values, (rows, cols)), shape=(np.max(rows) + 1, np.max(cols) + 1))


def load_test(path, name, sep, df_name, shape=None):
    df = pd.read_csv(path + name, sep=sep, header=None, names=df_name)
    rows, cols = df[df_name[0]], df[df_name[1]]
    values = np.ones_like(rows)

    if shape:
        return csr_matrix((values, (rows, cols)), shape=shape)
    else:
        return csr_matrix((values, (rows, cols)), shape=(np.max(rows) + 1, np.max(cols) + 1))


def load_rating(path, name, sep, df_name, shape=None):
    df = pd.read_csv(path + name, sep=sep, header=None, names=df_name)
    rows, cols, values = df[df_name[0]], df[df_name[1]], df[df_name[2]]

    if shape:
        return csr_matrix((values, (rows, cols)), shape=shape)
    else:
        return csr_matrix((values, (rows, cols)), shape=(np.max(rows) + 1, np.max(cols) + 1))


def save_numpy(matrix, path, model):
    save_npz('{0}{1}'.format(path, model), matrix)


def load_numpy(path, name):
    return load_npz(path+name).tocsr()


def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)[key]
        except yaml.YAMLError as exc:
            print(exc)


def load_dataframe_csv(path, name):
    return pd.read_csv(path+name)


def save_dataframe_csv(df, path, name):
    df.to_csv(path+name, index=False)


def find_best_hyperparameters(folder_path, meatric):
    csv_files = [join(folder_path, f) for f in listdir(folder_path)
                 if isfile(join(folder_path, f)) and f.endswith('tuning.csv') and not f.startswith('final')]
    best_settings = []
    for record in csv_files:
        df = pd.read_csv(record)
        df[meatric+'_Score'] = df[meatric].map(lambda x: literal_eval(x)[0])
        best_settings.append(df.loc[df[meatric+'_Score'].idxmax()].to_dict())

    df = pd.DataFrame(best_settings).drop(meatric+'_Score', axis=1)

    return df