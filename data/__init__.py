import os

import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from data.dataset import SYSUDataset
from data.dataset import RegDBDataset
from data.dataset import MarketDataset

from data.sampler import CrossModalityIdentitySampler
from data.sampler import CrossModalityRandomSampler
from data.sampler import RandomIdentitySampler
from data.sampler import NormTripletSampler


def collate_fn(batch):  # img, label, cam_id, img_path, img_id
    samples = list(zip(*batch))

    data = [torch.stack(x, 0) for i, x in enumerate(samples) if i != 3]
    data.insert(3, samples[3])
    return data


def get_train_loader(dataset, root, sample_method, batch_size, p_size, k_size, image_size, random_flip=False, random_crop=False,
                     random_erase=False, color_jitter=False, padding=0, num_workers=4):
    # data pre-processing
    t = [T.Resize(image_size)]

    if random_flip:
        t.append(T.RandomHorizontalFlip())

    if color_jitter:
        t.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))

    if random_crop:
        t.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])

    t.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if random_erase:
        t.append(T.RandomErasing())
        # t.append(Jigsaw())

    transform = T.Compose(t)

    # dataset
    if dataset == 'sysu':
        train_dataset = SYSUDataset(root, mode='train', transform=transform)
    elif dataset == 'regdb':
        train_dataset = RegDBDataset(root, mode='train', transform=transform)
    elif dataset == 'market':
        train_dataset = MarketDataset(root, mode='train', transform=transform)

    # sampler
    assert sample_method in ['random', 'identity_uniform', 'identity_random', 'norm_triplet']
    if sample_method == 'identity_uniform':
        batch_size = p_size * k_size
        sampler = CrossModalityIdentitySampler(train_dataset, p_size, k_size)
    elif sample_method == 'identity_random':
        batch_size = p_size * k_size
        sampler = RandomIdentitySampler(train_dataset, p_size * k_size, k_size)
    elif sample_method == 'norm_triplet':
        batch_size = p_size * k_size
        sampler = NormTripletSampler(train_dataset, p_size * k_size, k_size)
    else:
        sampler = CrossModalityRandomSampler(train_dataset, batch_size)

    # loader
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, drop_last=True, pin_memory=True,
                              collate_fn=collate_fn, num_workers=num_workers)

    return train_loader


def get_test_loader(dataset, root, batch_size, image_size, num_workers=4):
    # transform
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dataset
    if dataset == 'sysu':
        gallery_dataset = SYSUDataset(root, mode='gallery', transform=transform)
        query_dataset = SYSUDataset(root, mode='query', transform=transform)
    elif dataset == 'regdb':
        gallery_dataset = RegDBDataset(root, mode='gallery', transform=transform)
        query_dataset = RegDBDataset(root, mode='query', transform=transform)
    elif dataset == 'market':
        gallery_dataset = MarketDataset(root, mode='gallery', transform=transform)
        query_dataset = MarketDataset(root, mode='query', transform=transform)

    # dataloader
    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    gallery_loader = DataLoader(dataset=gallery_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn,
                                num_workers=num_workers)

    return gallery_loader, query_loader
