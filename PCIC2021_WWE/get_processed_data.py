import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

from utils.progress import WorkSplitter
from utils.io import load_bigtag, load_choicetag, load_valid, load_test, load_rating, save_numpy


def main(args):
    progress = WorkSplitter()
    progress.section("Generate User-Tag Matrix")
    file_path = open('E:/PycharmProjects/PCIC21_Causal/datax/train/item2tag.json', 'r')
    item2tag = json.load(file_path)

    bigtag_matrix = load_bigtag(path=args.path, name='train/' + args.big, sep=args.sep, df_name=['uid', 'iid', 'tag'],
                                item2tag=item2tag, shape=[1000, 1720])

    choicetag_matrix = load_choicetag(path=args.path, name='train/' + args.choice, sep=args.sep,
                                      df_name=['uid', 'iid', 'tag'], item2tag=item2tag, shape=bigtag_matrix.shape,
                                      filter_matrix=bigtag_matrix)

    valid_matrix = load_valid(path=args.path, name='valid/' + args.valid, sep=args.sep, df_name=['uid', 'iid', 'tag'],
                              shape=bigtag_matrix.shape)

    test_matrix = load_test(path=args.path, name='test/' + args.test, sep=args.sep, df_name=['uid', 'iid'],
                            shape=bigtag_matrix.shape)

    test2_matrix = load_test(path=args.path, name='test/' + args.test2, sep=args.sep, df_name=['uid', 'iid'],
                             shape=bigtag_matrix.shape)

    rating_matrix = load_rating(path=args.path, name='train/' + args.rating, sep=args.sep,
                                df_name=['uid', 'iid', 'rating'], shape=[1000, 1000])

    save_dir = args.path + 'processed/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_numpy(bigtag_matrix, save_dir, "user_big_mat")
    save_numpy(choicetag_matrix, save_dir, "user_choice_mat")
    save_numpy(valid_matrix, save_dir, "valid")
    save_numpy(test_matrix, save_dir, "test")
    save_numpy(test2_matrix, save_dir, "test2")
    save_numpy(rating_matrix, save_dir, "rating")

    print('* user_big  #num: %6d, pos: %.6f, neg: %.6f' % (
        bigtag_matrix.count_nonzero(), np.sum(bigtag_matrix == 1) / bigtag_matrix.count_nonzero(),
        1 - np.sum(bigtag_matrix == 1) / bigtag_matrix.count_nonzero()))
    print('* user_choice  #num: %6d, pos: %.6f, neg: %.6f' % (
        choicetag_matrix.count_nonzero(), np.sum(choicetag_matrix == 1) / choicetag_matrix.count_nonzero(),
        1 - np.sum(choicetag_matrix == 1) / choicetag_matrix.count_nonzero()))
    print('* valid  #num: %6d, pos: %.6f, neg: %.6f' % (
        valid_matrix.count_nonzero(), np.sum(valid_matrix == 1) / valid_matrix.count_nonzero(),
        1 - np.sum(valid_matrix == 1) / valid_matrix.count_nonzero()))

    # progress.section('正负性和tag的关系')
    # ui_pairs = lil_matrix(bigtag_matrix)
    # ui_pairs = np.asarray(ui_pairs.nonzero()).T.astype('int32')
    # label = np.asarray(bigtag_matrix[ui_pairs[:, 0], ui_pairs[:, 1]]).T
    #
    # _ui_pairs = lil_matrix(choicetag_matrix)
    # _ui_pairs = np.asarray(_ui_pairs.nonzero()).T.astype('int32')
    # _label = np.asarray(choicetag_matrix[_ui_pairs[:, 0], _ui_pairs[:, 1]]).T
    #
    # all_tag = np.hstack((np.vstack((ui_pairs, _ui_pairs)), np.vstack((label, _label))))
    # all_tag = np.unique(all_tag, axis=0)
    #
    # num_tag = 1720
    # tag_idx = np.arange(num_tag)
    # tag_frequency = np.zeros_like(tag_idx)
    # for i in range(all_tag.shape[0]):
    #     tag_frequency[all_tag[i, 1]] += 1
    # tag_frequency_mat = np.hstack((np.expand_dims(tag_idx, 1), np.expand_dims(tag_frequency, 1)))
    # index = np.argsort(-tag_frequency_mat[:, -1])
    # tag_frequency_mat = tag_frequency_mat[index]
    #
    # pos_tag_dict = dict()
    # neg_tag_dict = dict()
    # for i in range(all_tag.shape[0]):
    #     if all_tag[i, 2] == 1:
    #         if all_tag[i, 1] not in pos_tag_dict:
    #             pos_tag_dict[all_tag[i, 1]] = 1
    #         else:
    #             pos_tag_dict[all_tag[i, 1]] += 1
    #
    #     if all_tag[i, 2] == -1:
    #         if all_tag[i, 1] not in neg_tag_dict:
    #             neg_tag_dict[all_tag[i, 1]] = 1
    #         else:
    #             neg_tag_dict[all_tag[i, 1]] += 1
    #
    # sort_pos_tag = np.zeros_like(tag_frequency_mat).astype(float)
    # for i in range(tag_frequency_mat.shape[0]):
    #     sort_pos_tag[i, 0] = tag_frequency_mat[i, 0]
    #     if tag_frequency_mat[i, 0] not in pos_tag_dict:
    #         sort_pos_tag[i, 1] = np.nan
    #     else:
    #         sort_pos_tag[i, 1] = pos_tag_dict[tag_frequency_mat[i, 0]]
    #
    # sort_neg_tag = np.zeros_like(tag_frequency_mat).astype(float)
    # for i in range(tag_frequency_mat.shape[0]):
    #     sort_neg_tag[i, 0] = tag_frequency_mat[i, 0]
    #     if tag_frequency_mat[i, 0] not in neg_tag_dict:
    #         sort_neg_tag[i, 1] = np.nan
    #     else:
    #         sort_neg_tag[i, 1] = neg_tag_dict[tag_frequency_mat[i, 0]]
    #
    # plt.figure(0)
    # f, ax = plt.subplots()
    # plt.plot(np.arange(sort_pos_tag.shape[0]), sort_pos_tag[:, 1], 'ro--', linewidth=1)
    # # plt.plot(np.arange(sort_neg_tag.shape[0]), sort_neg_tag[:, 1], 'bv--', linewidth=1)
    # plt.show()
    #
    # # validation subset
    # ui_pairs = lil_matrix(valid_matrix)
    # ui_pairs = np.asarray(ui_pairs.nonzero()).T.astype('int32')
    # label = np.asarray(valid_matrix[ui_pairs[:, 0], ui_pairs[:, 1]]).T
    # all_tag = np.hstack((ui_pairs, label))
    #
    # pos_tag_dict = dict()
    # neg_tag_dict = dict()
    # for i in range(all_tag.shape[0]):
    #     if all_tag[i, 2] == 1:
    #         if all_tag[i, 1] not in pos_tag_dict:
    #             pos_tag_dict[all_tag[i, 1]] = 1
    #         else:
    #             pos_tag_dict[all_tag[i, 1]] += 1
    #
    #     if all_tag[i, 2] == -1:
    #         if all_tag[i, 1] not in neg_tag_dict:
    #             neg_tag_dict[all_tag[i, 1]] = 1
    #         else:
    #             neg_tag_dict[all_tag[i, 1]] += 1
    #
    # sort_pos_tag = np.zeros_like(tag_frequency_mat).astype(float)
    # for i in range(tag_frequency_mat.shape[0]):
    #     sort_pos_tag[i, 0] = tag_frequency_mat[i, 0]
    #     if tag_frequency_mat[i, 0] not in pos_tag_dict:
    #         sort_pos_tag[i, 1] = np.nan
    #     else:
    #         sort_pos_tag[i, 1] = pos_tag_dict[tag_frequency_mat[i, 0]]
    #
    # sort_neg_tag = np.zeros_like(tag_frequency_mat).astype(float)
    # for i in range(tag_frequency_mat.shape[0]):
    #     sort_neg_tag[i, 0] = tag_frequency_mat[i, 0]
    #     if tag_frequency_mat[i, 0] not in neg_tag_dict:
    #         sort_neg_tag[i, 1] = np.nan
    #     else:
    #         sort_neg_tag[i, 1] = neg_tag_dict[tag_frequency_mat[i, 0]]
    #
    # plt.figure(1)
    # f, ax = plt.subplots()
    # plt.plot(np.arange(sort_pos_tag.shape[0]), sort_pos_tag[:, 1], 'ro--', linewidth=1)
    # # plt.plot(np.arange(sort_neg_tag.shape[0]), sort_neg_tag[:, 1], 'bv--', linewidth=1)
    # plt.show()


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument('-p', dest='path', default='datax/')
    parser.add_argument('-r', dest='rating', help='rating subset', default='rating.txt')
    parser.add_argument('-b', dest='big', help='bigtag subset', default='bigtag.txt')
    parser.add_argument('-c', dest='choice', help='choicetag subset', default='choicetag.txt')
    parser.add_argument('-m', dest='movie', help='movie subset', default='movie.txt')
    parser.add_argument('-v', dest='valid', help='validation subset', default='validation.txt')
    parser.add_argument('-t', dest='test', help='test subset', default='test.txt')
    parser.add_argument('-t2', dest='test2', help='test subset 2', default='test_phase2.txt')
    parser.add_argument('-s', dest='sep', help='separate', default=' ')
    args = parser.parse_args()
    main(args)
