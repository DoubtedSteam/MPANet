import os
import random
import numpy as np
import scipy.io as sio
import matplotlib as mpl

mpl.use('AGG')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == '__main__':
    test_ids = [
        6, 10, 17, 21, 24, 25, 27, 28, 31, 34, 36, 37, 40, 41, 42, 43, 44, 45, 49, 50, 51, 54, 63, 69, 75, 80, 81, 82,
        83, 84, 85, 86, 87, 88, 89, 90, 93, 102, 104, 105, 106, 108, 112, 116, 117, 122, 125, 129, 130, 134, 138, 139,
        150, 152, 162, 166, 167, 170, 172, 176, 185, 190, 192, 202, 204, 207, 210, 215, 223, 229, 232, 237, 252, 253,
        257, 259, 263, 266, 269, 272, 273, 274, 275, 282, 285, 291, 300, 301, 302, 303, 307, 312, 315, 318, 331, 333
    ]
    random.seed(0)
    tsne = TSNE(n_components=2, init='pca')
    selected_ids = random.sample(test_ids, 20)
    plt.figure(figsize=(5, 5))

    # features without dual path
    q_mat_path = 'features/sysu/query-sysu-test-nodual-nore-adam-16x8-grey_model_150.mat'
    g_mat_path = 'features/sysu/gallery-sysu-test-nodual-nore-adam-16x8-grey_model_150.mat'

    mat = sio.loadmat(q_mat_path)
    q_feats = mat["feat"]
    q_ids = mat["ids"].squeeze()
    flag = np.in1d(q_ids, selected_ids)
    q_feats = q_feats[flag]

    mat = sio.loadmat(g_mat_path)
    g_feats = mat["feat"]
    g_ids = mat["ids"].squeeze()
    flag = np.in1d(g_ids, selected_ids)
    g_feats = g_feats[flag]

    embed = tsne.fit_transform(np.concatenate([q_feats, g_feats], axis=0))
    c = ['r'] * q_feats.shape[0] + ['b'] * g_feats.shape[0]
    # plt.subplot(1, 2, 1)
    plt.scatter(embed[:, 0], embed[:, 1], c=c)

    # # features with dual path
    # q_mat_path = 'features/sysu/query-sysu-test-dual-nore-separatelayer12-0.05_model_30.mat'
    # g_mat_path = 'features/sysu/gallery-sysu-test-dual-nore-separatelayer12-0.05_model_30.mat'
    #
    # mat = sio.loadmat(q_mat_path)
    # q_feats = mat["feat"]
    # q_ids = mat["ids"].squeeze()
    # flag = np.in1d(q_ids, selected_ids)
    # q_feats = q_feats[flag]
    #
    # mat = sio.loadmat(g_mat_path)
    # g_feats = mat["feat"]
    # g_ids = mat["ids"].squeeze()
    # flag = np.in1d(g_ids, selected_ids)
    # g_feats = g_feats[flag]
    #
    # embed = tsne.fit_transform(np.concatenate([q_feats, g_feats], axis=0))
    # c = ['r'] * q_feats.shape[0] + ['b'] * g_feats.shape[0]
    # plt.subplot(1, 2, 2)
    # plt.scatter(embed[:, 0], embed[:, 1], c=c)

    plt.tight_layout()
    plt.savefig('tsne-adv-layer2-separate-l2.jpg')
