import math
import numpy as np
from tqdm import tqdm
import optuna
import random
from optuna.samplers import TPESampler
from optuna.trial import Trial
from utils.progress import WorkSplitter
from scipy.sparse import lil_matrix, csr_matrix
import torch
import torch.nn as nn
from torch.nn.init import normal_
from models.loss import TMFPLoss
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from ismember import ismember
from utils.io import load_numpy



class Objective:

    def __init__(self, num_users, num_tags, train, valid, test, rating, item2tag, pair_samples, iters, seed,
                 save) -> None:
        """Initialize Class"""
        self.num_users = num_users
        self.num_tags = num_tags
        self.train = train
        self.valid = valid
        self.test = test
        self.rating = rating
        self.item2tag = item2tag
        self.pair_samples = pair_samples
        self.iters = iters
        self.seed = seed
        self.save = save

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""

        # sample a set of hyperparameters.
        #rank = trial.suggest_discrete_uniform('rank', 4, 64, 4)
        #lam = trial.suggest_categorical('lambda', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        #batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024, 2048])
        #lr = trial.suggest_categorical('learning_rate', [0.001, 0.005, 0.01, 0.05, 0.1])
        #alpha = trial.suggest_uniform('alpha', 0.001, 0.1)
        #gamma = trial.suggest_uniform('gamma', 0.001, 0.99)

        rank = trial.suggest_categorical('rank', [64])
        lam = trial.suggest_categorical('lambda', [0.0001])
        batch_size = trial.suggest_categorical('batch_size', [128])
        lr = trial.suggest_categorical('learning_rate', [0.005])
        alpha = trial.suggest_categorical('alpha', [0.023844909983539674])
        gamma = trial.suggest_categorical('gamma', [0.0013034371963566707])

        setup_seed(self.seed)

        model = TMFP(self.num_users, self.num_tags, np.int(rank), np.int(batch_size), lamb=lam, learning_rate=lr,
                     alpha=alpha, gamma=gamma).cuda()

        score, _, _, _, _ = model.fit(self.train, self.valid, self.test, self.rating, self.item2tag, self.pair_samples,
                                      self.iters, self.seed, self.save)

        return score


class Tuner:
    """Class for tuning hyperparameter of MF models."""

    def __init__(self):
        """Initialize Class."""

    def tune(self, n_trials, num_users, num_tags, train, valid, test, rating, item2tag, pair_samples, num_epoch, seed,
             save):
        """Hyperparameter Tuning by TPE."""
        objective = Objective(num_users=num_users, num_tags=num_tags, train=train, valid=valid, test=test,
                              rating=rating, item2tag=item2tag, pair_samples=pair_samples, iters=num_epoch, seed=seed,
                              save=save)
        study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.trials_dataframe(), study.best_params


class TMFP(nn.Module):
    def __init__(self, num_users, num_tags, embed_dim, batch_size,
                 num_items=1000,
                 lamb=0.01,
                 learning_rate=1e-3,
                 alpha=0.1,
                 gamma=0.1,
                 **unused):
        super(TMFP, self).__init__()
        self.num_users = num_users
        self.num_tags = num_tags
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.lamb = lamb
        self.lr = learning_rate
        self.alpha = alpha
        self.gamma = gamma

        # Variable to learn
        self.user_e = nn.Embedding(self.num_users, self.embed_dim)
        self.tag_e = nn.Embedding(self.num_tags, self.embed_dim)
        self.user_b = nn.Embedding(self.num_users, 1)
        self.tag_b = nn.Embedding(self.num_tags, 1)
        self.item_q = nn.Embedding(self.num_items, 1)

        self.apply(self._init_weights)

        self.loss = TMFPLoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.1)

    def forward(self, user, tag):
        user_embedding = self.user_e(user)
        tag_embedding = self.tag_e(tag)

        preds = self.user_b(user)
        preds += self.tag_b(tag)
        preds += (user_embedding * tag_embedding).sum(dim=1, keepdim=True)

        return preds.squeeze()

    def pair_forward(self, user, pos_tag, neg_tag):
        user_embedding = self.user_e(user)
        pos_tag_embedding = self.tag_e(pos_tag)
        neg_tag_embedding = self.tag_e(neg_tag)

        pos_preds = self.user_b(user)
        pos_preds += self.tag_b(pos_tag)
        pos_preds += (user_embedding * pos_tag_embedding).sum(dim=1, keepdim=True)

        neg_preds = self.user_b(user)
        neg_preds += self.tag_b(neg_tag)
        neg_preds += (user_embedding * neg_tag_embedding).sum(dim=1, keepdim=True)

        return pos_preds.squeeze(), neg_preds.squeeze()

    def rating_forward(self, user, item, tag_index):
        user_embedding = self.user_e(user)
        all_tag_embedding = self.tag_e(tag_index)
        item_embedding = torch.mean(all_tag_embedding, dim=1)

        preds = self.user_b(user)
        a = self.tag_b(tag_index)
        preds += torch.mean(self.tag_b(tag_index), dim=1)
        preds += (user_embedding * item_embedding).sum(dim=1, keepdim=True)
        preds += self.item_q(item)

        return preds.squeeze()

    def calculate_loss(self, tag_user_list, tag_list, tag_label_list, rating_user_list, item_list, tag_index_list,
                       rating_label_list, pair_user_list, pos_tag_list, neg_tag_list):
        pair_pos_output, pair_neg_output = self.pair_forward(pair_user_list, pos_tag_list, neg_tag_list)

        return self.loss.forward(self.forward(tag_user_list, tag_list), tag_label_list,
                                 self.rating_forward(rating_user_list, item_list, tag_index_list), rating_label_list,
                                 self.alpha, pair_pos_output, pair_neg_output, self.gamma)

    def predict(self, user, tag):
        return self.forward(user, tag)

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.lamb)

    def get_embedding(self):
        return self.user_e, self.tag_e, self.user_b, self.tag_b

    def get_tag_index(self, ui_pairs, item2tag):
        tag_index = np.zeros((ui_pairs.shape[0], 8))
        for i in range(ui_pairs.shape[0]):
            temp = item2tag[str(ui_pairs[i, 1])]
            tag_index[i, :] = np.array(temp)
        return tag_index

    def fit(self, matrix_train, matrix_valid, matrix_test, matrix_rating, item2tag, pair_samples, num_epoch=100, seed=0,
            save=False):
        setup_seed(seed)

        optimizer = self.get_optimizer()

        # Load data
        tag_ui_pairs = lil_matrix(matrix_train)
        tag_ui_pairs = np.asarray(tag_ui_pairs.nonzero()).T.astype('int32')
        tag_train_label = np.asarray(matrix_train[tag_ui_pairs[:, 0], tag_ui_pairs[:, 1]]).T
        tag_train_label[tag_train_label == -1] = 0

        rating_ui_pairs = lil_matrix(matrix_rating)
        rating_ui_pairs = np.asarray(rating_ui_pairs.nonzero()).T.astype('int32')
        rating_train_label = np.asarray(matrix_rating[rating_ui_pairs[:, 0], rating_ui_pairs[:, 1]]).T
        tag_index = self.get_tag_index(rating_ui_pairs, item2tag)

        valid_ui_pairs = lil_matrix(matrix_valid)
        valid_ui_pairs = np.asarray(valid_ui_pairs.nonzero()).T.astype('int32')
        valid_label = np.asarray(matrix_valid[valid_ui_pairs[:, 0], valid_ui_pairs[:, 1]])[0]
        valid_label[valid_label == -1] = 0

        # Training
        tag_train_dataloader = DataLoader(np.hstack((tag_ui_pairs, tag_train_label)), self.batch_size, shuffle=True)
        pair_train_dataloader = DataLoader(pair_samples, math.ceil(pair_samples.shape[0]/len(tag_train_dataloader)),
                                           shuffle=True)
        rating_train_dataloader = DataLoader(np.hstack((rating_ui_pairs, rating_train_label, tag_index)),
                                             math.ceil(rating_ui_pairs.shape[0]/len(tag_train_dataloader)),
                                             shuffle=True)
        result, best_result, early_stop, best_U, best_V, best_uB, best_vB = 0, 0, 0, None, None, None, None
        for epoch in tqdm(range(num_epoch)):
            dataloader_iterator_1 = iter(tag_train_dataloader)
            dataloader_iterator_2 = iter(pair_train_dataloader)

            for i, rating_data in enumerate(rating_train_dataloader):
                tag_data = next(dataloader_iterator_1)
                pair_data = next(dataloader_iterator_2)

                tag_user = tag_data[:, 0].cuda()
                tag_item = tag_data[:, 1].cuda()
                tag_label = tag_data[:, 2].cuda()

                rating_user = rating_data[:, 0].cuda()
                rating_item = rating_data[:, 1].cuda()
                rating_label = rating_data[:, 2].cuda()
                tag_index = rating_data[:, 3:].cuda()

                pair_user = pair_data[:, 0].cuda()
                pair_pos_tag = pair_data[:, 1].cuda()
                pair_neg_tag = pair_data[:, 2].cuda()

                loss = self.calculate_loss(tag_user.long(), tag_item.long(), tag_label.float(), rating_user.long(),
                                           rating_item.long(), tag_index.long(), rating_label.float(),
                                           pair_user.long(), pair_pos_tag.long(), pair_neg_tag.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate
            # train_user = torch.LongTensor(tag_ui_pairs[:, 0]).cuda()
            # train_item = torch.LongTensor(tag_ui_pairs[:, 1]).cuda()
            # train_pred = self.predict(train_user, train_item)
            # train_result = roc_auc_score(tag_train_label.flatten(), train_pred.detach().cpu().numpy())

            valid_user = torch.LongTensor(valid_ui_pairs[:, 0]).cuda()
            valid_item = torch.LongTensor(valid_ui_pairs[:, 1]).cuda()
            idx1, idx2 = ismember(tag_ui_pairs, valid_ui_pairs, 'rows')
            valid_pred = self.predict(valid_user, valid_item)
            valid_pred = valid_pred.detach().cpu().numpy()
            # valid_pred[idx2] = tag_train_label.flatten()[idx1]
            #
            # valid_result = roc_auc_score(valid_label, valid_pred)

            _valid_label = np.delete(valid_label, idx2)
            _valid_pred = np.delete(valid_pred, idx2)
            valid_result = roc_auc_score(_valid_label, _valid_pred)

            if valid_result > best_result:
                # result = train_result
                best_result = valid_result
                embed_U, embed_V, embed_uB, embed_vB = self.get_embedding()
                best_U, best_V, best_uB, best_vB = embed_U.weight.detach().cpu().numpy(), \
                                                   embed_V.weight.detach().cpu().numpy(), \
                                                   embed_uB.weight.detach().cpu().numpy(), \
                                                   embed_vB.weight.detach().cpu().numpy(),
                early_stop = 0

                if save:
                    test_ui_pairs = lil_matrix(matrix_test)
                    test_ui_pairs = np.asarray(test_ui_pairs.nonzero()).T.astype('int32')

                    test_user = torch.LongTensor(test_ui_pairs[:, 0]).cuda()
                    test_item = torch.LongTensor(test_ui_pairs[:, 1]).cuda()
                    idx1, idx2 = ismember(tag_ui_pairs, test_ui_pairs, 'rows')
                    test_pred = self.predict(test_user, test_item)
                    test_pred = test_pred.detach().cpu().numpy()

                    test_pred = (test_pred - np.min(test_pred)) / (np.max(test_pred) - np.min(test_pred))
                    test_pred[idx2] = tag_train_label.flatten()[idx1]

                    test_pred = test_pred.reshape(-1, 1)
                    submit = np.hstack((test_ui_pairs, test_pred))
                    np.savetxt("./tables/submit.csv", submit, fmt=('%d', '%d', '%f'))

            else:
                early_stop += 1
                if early_stop > 5:
                    break
        print('training set AUC is {0}'.format(result))

        return best_result, best_U, best_V, best_uB, best_vB


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def tmfp(matrix_train, matrix_valid, matrix_test, matrix_rating, item2tag, matrix_utrain=None, pair_samples=None,
         iteration=100, lam=0.01, rank=200, batch_size=500, learning_rate=1e-3, seed=0, alpha=0.1, gamma=0.1,
         source=None, searcher='grid', n_trials=1, save=True, **unused):
    progress = WorkSplitter()
 
    progress.section("TMF-Pair: Set the random seed")
    setup_seed(seed)
    pesudo_tagging = load_numpy(path='datax/', name="processed/" + 'test1_pesudo.npz')


    progress.section("TMF-Pair: Training")
    if source == "unif":  # Source of training data: logged data (None), uniform data ("unif") and both ("combine")
        matrix_train = matrix_utrain

    elif source == "combine":
        ui_pairs = lil_matrix(matrix_train)
        ui_pairs = np.asarray(ui_pairs.nonzero()).T.astype('int32')
        label = np.asarray(matrix_train[ui_pairs[:, 0], ui_pairs[:, 1]]).T

        _ui_pairs_p = lil_matrix(pesudo_tagging)
        _ui_pairs_p = np.asarray(_ui_pairs_p.nonzero()).T.astype('int32')
        _label_p = np.asarray(pesudo_tagging[_ui_pairs_p[:, 0], _ui_pairs_p[:, 1]]).T

        _ui_pairs = lil_matrix(matrix_utrain)
        _ui_pairs = np.asarray(_ui_pairs.nonzero()).T.astype('int32')
        _label = np.asarray(matrix_utrain[_ui_pairs[:, 0], _ui_pairs[:, 1]]).T

        combine_data = np.hstack((np.vstack((ui_pairs, _ui_pairs, _ui_pairs_p)), np.vstack((label, _label, _label_p))))
        combine_data = np.unique(combine_data, axis=0)

        matrix_train = csr_matrix((combine_data[:, 2], (combine_data[:, 0], combine_data[:, 1])),
                                  shape=matrix_train.shape)

    matrix_input = matrix_train

    m, n = matrix_input.shape

    if searcher == 'optuna':
        tuner = Tuner()
        trials, best_params = tuner.tune(n_trials=n_trials, num_users=m, num_tags=n, train=matrix_input,
                                         valid=matrix_valid, test=matrix_test, rating=matrix_rating, item2tag=item2tag,
                                         pair_samples=pair_samples, num_epoch=iteration, seed=seed, save=save)
        return trials, best_params

    if searcher == 'grid':
        model = TMFP(m, n, rank, batch_size, lamb=lam, learning_rate=learning_rate, alpha=alpha, gamma=gamma).cuda()

        _, U, V, uB, vB = model.fit(matrix_input, matrix_valid, matrix_test, matrix_rating, item2tag, pair_samples,
                                    iteration, seed)

        return U, V, uB, vB