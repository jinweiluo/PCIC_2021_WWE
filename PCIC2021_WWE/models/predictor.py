import numpy as np
from scipy.sparse import lil_matrix


def predict(matrix_U, matrix_V, matrix_Test, bias_U=None, bias_V=None, gpu=True):
    if gpu:
        import cupy as cp
        matrix_U = cp.array(matrix_U)
        matrix_V = cp.array(matrix_V)

    user_item_matrix = lil_matrix(matrix_Test)
    user_item_pairs = np.asarray(user_item_matrix.nonzero()).T

    rating_predict = rating_sub_routine(matrix_U, matrix_V, user_item_pairs, bias_U, bias_V, gpu=gpu)

    return rating_predict


def rating_sub_routine(matrix_U, matrix_V, user_item_pairs, bias_U, bias_V, gpu=True):
    temp_U = matrix_U[user_item_pairs[:, 0], :]
    temp_V = matrix_V[user_item_pairs[:, 1], :]

    if gpu:
        import cupy as cp
        vector_predict = cp.sum(temp_U * temp_V, axis=1)
    else:
        vector_predict = np.sum(temp_U * temp_V, axis=1)

    if bias_U is not None:
        if gpu:
            import cupy as cp
            temp_bias = bias_U[user_item_pairs[:, 0]]
            vector_predict = vector_predict + cp.array(temp_bias)
        else:
            temp_bias = bias_U[user_item_pairs[:, 0]]
            vector_predict = vector_predict + temp_bias

    if bias_V is not None:
        if gpu:
            import cupy as cp
            temp_bias = bias_V[user_item_pairs[:, 1]]
            vector_predict = vector_predict + cp.array(temp_bias)
        else:
            temp_bias = bias_V[user_item_pairs[:, 1]]
            vector_predict = vector_predict + temp_bias

    if gpu:
        import cupy as cp
        vector_predict = cp.asnumpy(vector_predict)
    return vector_predict
