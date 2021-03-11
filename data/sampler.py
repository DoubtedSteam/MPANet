import numpy as np

import copy
from torch.utils.data import Sampler
from collections import defaultdict


class CrossModalityRandomSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.rgb_list = []
        self.ir_list = []
        for i, cam in enumerate(dataset.cam_ids):
            if cam in [3, 6]:
                self.ir_list.append(i)
            else:
                self.rgb_list.append(i)

    def __len__(self):
        return max(len(self.rgb_list), len(self.ir_list)) * 2

    def __iter__(self):
        sample_list = []
        rgb_list = np.random.permutation(self.rgb_list).tolist()
        ir_list = np.random.permutation(self.ir_list).tolist()

        rgb_size = len(self.rgb_list)
        ir_size = len(self.ir_list)
        if rgb_size >= ir_size:
            diff = rgb_size - ir_size
            reps = diff // ir_size
            pad_size = diff % ir_size
            for _ in range(reps):
                ir_list.extend(np.random.permutation(self.ir_list).tolist())
            ir_list.extend(np.random.choice(self.ir_list, pad_size, replace=False).tolist())
        else:
            diff = ir_size - rgb_size
            reps = diff // ir_size
            pad_size = diff % ir_size
            for _ in range(reps):
                rgb_list.extend(np.random.permutation(self.rgb_list).tolist())
            rgb_list.extend(np.random.choice(self.rgb_list, pad_size, replace=False).tolist())

        assert len(rgb_list) == len(ir_list)

        half_bs = self.batch_size // 2
        for start in range(0, len(rgb_list), half_bs):
            sample_list.extend(rgb_list[start:start + half_bs])
            sample_list.extend(ir_list[start:start + half_bs])

        return iter(sample_list)


class CrossModalityIdentitySampler(Sampler):
    def __init__(self, dataset, p_size, k_size):
        self.dataset = dataset
        self.p_size = p_size
        self.k_size = k_size // 2
        self.batch_size = p_size * k_size * 2

        self.id2idx_rgb = defaultdict(list)
        self.id2idx_ir = defaultdict(list)
        for i, identity in enumerate(dataset.ids):
            if dataset.cam_ids[i] in [3, 6]:
                self.id2idx_ir[identity].append(i)
            else:
                self.id2idx_rgb[identity].append(i)

    def __len__(self):
        return self.dataset.num_ids * self.k_size * 2

    def __iter__(self):
        sample_list = []

        id_perm = np.random.permutation(self.dataset.num_ids)
        for start in range(0, self.dataset.num_ids, self.p_size):
            selected_ids = id_perm[start:start + self.p_size]

            sample = []
            for identity in selected_ids:
                replace = len(self.id2idx_rgb[identity]) < self.k_size
                s = np.random.choice(self.id2idx_rgb[identity], size=self.k_size, replace=replace)
                sample.extend(s)

            sample_list.extend(sample)

            sample.clear()
            for identity in selected_ids:
                replace = len(self.id2idx_ir[identity]) < self.k_size
                s = np.random.choice(self.id2idx_ir[identity], size=self.k_size, replace=replace)
                sample.extend(s)

            sample_list.extend(sample)

        return iter(sample_list)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic_R = defaultdict(list)
        self.index_dic_I = defaultdict(list)
        for i, identity in enumerate(data_source.ids):
            if data_source.cam_ids[i] in [3, 6]:
                self.index_dic_I[identity].append(i)
            else:
                self.index_dic_R[identity].append(i)
        self.pids = list(self.index_dic_I.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic_I[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs_I = copy.deepcopy(self.index_dic_I[pid])
            idxs_R = copy.deepcopy(self.index_dic_R[pid])
            if len(idxs_I) < self.num_instances // 2 and len(idxs_R) < self.num_instances // 2:
                idxs_I = np.random.choice(idxs_I, size=self.num_instances // 2, replace=True)
                idxs_R = np.random.choice(idxs_R, size=self.num_instances // 2, replace=True)
            if len(idxs_I) > len(idxs_R):
                idxs_I = np.random.choice(idxs_I, size=len(idxs_R), replace=False)
            if len(idxs_R) > len(idxs_I):
                idxs_R = np.random.choice(idxs_R, size=len(idxs_I), replace=False)
            np.random.shuffle(idxs_I)
            np.random.shuffle(idxs_R)
            batch_idxs = []
            for idx_I, idx_R in zip(idxs_I, idxs_R):
                batch_idxs.append(idx_I)
                batch_idxs.append(idx_R)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class NormTripletSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(self.data_source.ids):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length