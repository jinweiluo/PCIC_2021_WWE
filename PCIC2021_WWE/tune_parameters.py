import numpy as np
import json
import argparse
from utils.modelnames import models
from utils.io import load_numpy, load_yaml
from utils.argcheck import check_int_positive
from experiment.tuning import hyper_parameter_tuning


def main(args):
    params = load_yaml(args.grid)
    params['models'] = {params['models']: models[params['models']]}
    R_train = load_numpy(path=args.path, name=args.document + args.train)
    R_valid = load_numpy(path=args.path, name=args.document + args.valid)
    R_test = load_numpy(path=args.path, name=args.document + args.test)
    R_rtrain = load_numpy(path=args.path, name=args.document + args.unif_train)
    R_rating = load_numpy(path=args.path, name=args.document + args.rating)
    item2tag = json.load(open(args.path + 'train/' + args.item2tag, 'r'))
    pair_samples = np.load(args.path + args.document + args.pair)
    hyper_parameter_tuning(R_train, R_valid, R_test, R_rtrain, R_rating, item2tag, pair_samples, params,
                           source=args.source, seed=args.seed, searcher=args.searcher, save_path=args.table_name,
                           op_save_path=args.op_table_name)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")
    parser.add_argument('-tb', dest='table_name', default="tmfp111_tuning.csv")
    parser.add_argument('-otb', dest='op_table_name', default="op_tmfp1111_tuning.csv")
    parser.add_argument('-p', dest='path', default="datax/")
    parser.add_argument('-d', dest='document', default="processed/")
    parser.add_argument('-t', dest='train', default='user_big_mat.npz')
    parser.add_argument('-v', dest='valid', default='valid.npz')
    parser.add_argument('-te', dest='test', default='test2.npz')
    parser.add_argument('-ut', dest='unif_train', default='user_choice_mat.npz')
    parser.add_argument('-r', dest='rating', default='rating.npz')
    parser.add_argument('-it', dest='item2tag', default='item2tag.json')
    parser.add_argument('-pa', dest='pair', default='pair_samples.npy')
    parser.add_argument('-sr', dest='source', default='combine')  # or 'unif' or 'combine'
    parser.add_argument('-y', dest='grid', default='config/tmfp.yml')
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=2021)
    parser.add_argument('-searcher', dest='searcher', default="optuna")
    args = parser.parse_args()

    main(args)
