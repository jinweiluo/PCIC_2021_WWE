import json
from utils.progress import WorkSplitter
from numpy import genfromtxt
import numpy as np
from ismember import ismember
from utils.io import load_numpy
from scipy.sparse import lil_matrix


progress = WorkSplitter()
pre_compute = True

if pre_compute:
    item2tag = dict()
    movie_file = open('./datax/train/movie.txt', 'r', encoding='utf-8')
    for line in movie_file.readlines():
        temp = line.strip().split(' ')
        item2tag[int(temp[0])] = list(map(int, temp[1:]))
    file_path = open('./datax/train/item2tag.json', 'w')
    json.dump(item2tag, file_path)
    file_path.close()

    user_tag = genfromtxt('./datax/train/bigtag.txt', delimiter=' ', dtype=int)
    user_item_tag_dict = dict()
    for i in range(user_tag.shape[0]):
        if user_tag[i, 2] != -1:
            if str(user_tag[i, 0]) + ',' + str(user_tag[i, 1]) not in user_item_tag_dict:
                user_item_tag_dict[str(user_tag[i, 0]) + ',' + str(user_tag[i, 1])] = {user_tag[i, 2]}
            else:
                user_item_tag_dict[str(user_tag[i, 0]) + ',' + str(user_tag[i, 1])].add(user_tag[i, 2])

    pair_samples = []
    for ui, tag in user_item_tag_dict.items():
        user_id, item_id = ui.split(',')[0], ui.split(',')[1]
        neg_tags = set(item2tag[int(item_id)]) - tag
        a = np.array(np.meshgrid(list(tag), list(neg_tags)))
        b = np.array(np.meshgrid(list(tag), list(neg_tags))).T
        temp = np.array(np.meshgrid(list(tag), list(neg_tags))).T.reshape(-1, 2)
        temp = np.hstack((int(user_id) * np.ones((temp.shape[0], 1)), temp))
        pair_samples.extend(temp.tolist())

    pair_samples = np.array(pair_samples)
    np.save('./datax/processed/pair_samples.npy', pair_samples)

progress.section('The total number of tags')
file_path = open('./datax/train/item2tag.json', 'r')
item2tag = json.load(file_path)
tag_set = set()
for i, t in item2tag.items():
    tag_set = tag_set.union(set(t))
print('The total number of tags is {}'.format(len(tag_set)))  # =1720

progress.section('rating set和tag set的交集')
user_rating = genfromtxt('./datax/train/rating.txt', delimiter=' ', dtype=int)
user_tag = genfromtxt('./datax/train/bigtag.txt', delimiter=' ', dtype=int)
user_choice_tag = genfromtxt('./datax/train/choicetag.txt', delimiter=' ', dtype=int)
_, index_1 = ismember(np.unique(user_rating[:, :2], axis=0), np.unique(user_tag[:, :2], axis=0), 'rows')
_, index_2 = ismember(np.unique(user_rating[:, :2], axis=0), np.unique(user_choice_tag[:, :2], axis=0), 'rows')
print('The number of rating set is {}'.format(user_rating.shape[0]))  # =19903
print('The number of tag set is {}'.format(user_tag.shape[0]))  # =8612
print('The number of choice-tag set is {}'.format(user_choice_tag.shape[0]))  # =1540
print('The number of intersections between the rating set and the tag set is {}'.format(
    index_1.shape[0]))  # =166
print('The number of intersections between the rating set and the choice-tag set is {}'.format(
    index_2.shape[0]))  # =21

progress.section('rating和tag之间的噪声')
_, index_1 = ismember(np.unique(user_rating[:, :2], axis=0), np.unique(user_tag[:, :2], axis=0), 'rows')
_, index_2 = ismember(np.unique(user_tag[:, :2], axis=0), np.unique(user_rating[:, :2], axis=0), 'rows')

_, index_3 = ismember(np.unique(user_rating[:, :2], axis=0), np.unique(user_choice_tag[:, :2], axis=0), 'rows')
_, index_4 = ismember(np.unique(user_choice_tag[:, :2], axis=0), np.unique(user_rating[:, :2], axis=0), 'rows')

confusion_11 = np.sum((user_rating[index_2, 2] > 3) & (user_tag[index_1, 2] == -1))
confusion_12 = np.sum((user_rating[index_4, 2] > 3) & (user_choice_tag[index_3, 2] == -1))

confusion_21 = np.sum((user_rating[index_2, 2] < 3) & (user_tag[index_1, 2] != -1))
confusion_22 = np.sum((user_rating[index_4, 2] < 3) & (user_choice_tag[index_3, 2] != -1))

# Don't like tags but have a high rating
print('The number of confusion 1 is {}, {}'.format(confusion_11, confusion_12))
# Like tags but have a low rating
print('The number of confusion 2 is {}, {}'.format(confusion_21, confusion_22))

progress.section('各个set之间的交集')
R_train = load_numpy(path='datax/', name="processed/user_big_mat.npz")
R_valid = load_numpy(path='datax/', name="processed/valid.npz")
R_test = load_numpy(path='datax/', name="processed/test.npz")
R_rtrain = load_numpy(path='datax/', name="processed/user_choice_mat.npz")

train_bigtag = lil_matrix(R_train)
train_bigtag = np.asarray(train_bigtag.nonzero()).T.astype('int32')
train_choicetag = lil_matrix(R_rtrain)
train_choicetag = np.asarray(train_choicetag.nonzero()).T.astype('int32')
valid_tag = lil_matrix(R_valid)
valid_tag = np.asarray(valid_tag.nonzero()).T.astype('int32')
test_tag = lil_matrix(R_test)
test_tag = np.asarray(test_tag.nonzero()).T.astype('int32')

_, inter_1 = ismember(np.unique(train_bigtag, axis=0), np.unique(train_choicetag, axis=0), 'rows')
_, inter_2 = ismember(np.unique(train_bigtag, axis=0), np.unique(valid_tag, axis=0), 'rows')
_, inter_3 = ismember(np.unique(train_bigtag, axis=0), np.unique(test_tag, axis=0), 'rows')
_, inter_4 = ismember(np.unique(train_choicetag, axis=0), np.unique(valid_tag, axis=0), 'rows')
_, inter_5 = ismember(np.unique(train_choicetag, axis=0), np.unique(test_tag, axis=0), 'rows')
_, inter_6 = ismember(np.unique(valid_tag, axis=0), np.unique(test_tag, axis=0), 'rows')
print('The overlap of experimental data is {}, {}, {}, {}, {}, {}'.format(inter_1.shape[0], inter_2.shape[0], inter_3.shape[0],
                                                                          inter_4.shape[0], inter_5.shape[0], inter_6.shape[0]))
