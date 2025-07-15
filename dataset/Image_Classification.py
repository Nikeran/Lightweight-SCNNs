import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from collections import defaultdict
import random
import pickle
from PIL import Image

def load_cifar10(data_dir, train=True, img_size=32):
	"""
	Loads CIFAR-10 as two concatenated tensors.

	Args:
		data_dir (str): path to the 'cifar-10-batches-py' folder
		train (bool): if True, loads all 5 training batches;
					  if False, loads the single test batch.

	Returns:
		images (Tensor): shape [N, 3, 32, 32], float32 scaled to [0,1]
		labels (LongTensor): shape [N]
	"""
	# choose files
	if train:
		batch_files = [f"data_batch_{i}" for i in range(1, 6)]
	else:
		batch_files = ["test_batch"]

	all_imgs = []
	all_lbls = []

	for fname in batch_files:
		path = os.path.join(data_dir, fname)
		with open(path, 'rb') as f:
			batch = pickle.load(f, encoding='latin1')

		# reshape and normalize
		data = batch['data']                    # numpy (10000, 3072)
		data = data.reshape(-1, 3, img_size, img_size)      # (10000, 3, 32, 32)
		data = torch.from_numpy(data).float() / 255.0

		labels = torch.tensor(batch['labels'], dtype=torch.long)

		all_imgs.append(data)
		all_lbls.append(labels)

	images = torch.cat(all_imgs, dim=0)   # e.g. (50000, 3, 32, 32) if train
	labels = torch.cat(all_lbls, dim=0)   # e.g. (50000,)

	return images, labels


class ImgClassificationDataset(Dataset):
	"""
	Generic image classification dataset class from data.
	3D RGB images uint8, labels as integers.
	"""
	def __init__(self, data, labels, transform=None):
		self.data = data
		self.labels = labels
		self.transform = transform

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		img = self.data[idx]

		if isinstance(img, np.ndarray):
			img = transforms.ToPILImage()(img)
		img = img.convert("RGB")
		if self.transform:
			img = self.transform(img)

		
		label = int(self.labels[idx])
		return img, label


class ImgClassificationDatasetHF(Dataset):
    """
    Wrapper for Img class using a Hugging Face Dataset object.
    Expects dataset items to be dicts with keys: 'image', 'label'
    """
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image']  # PIL.Image
        label = item['label']

        if self.transform:
            img = self.transform(img)

        return img, label


class CIFAR10C(Dataset):
	"""
	CIFAR-10-C corruptions dataset loader.
	Expects .npy files: data_dir/<corruption>.npy and labels.npy
	"""
	def __init__(self, data_dir, corruption='gaussian_noise', severity=1, transform=None):
		arr = np.load(os.path.join(data_dir, f"{corruption}.npy"))
		self.data = arr[(severity-1)*10000:severity*10000]
		all_labels = np.load(os.path.join(data_dir, 'labels.npy'))
		self.labels = all_labels[(severity-1)*10000:severity*10000]
		self.transform = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		img = self.data[idx]
		img = transforms.ToPILImage()(img)
		if self.transform:
			img = self.transform(img)
		label = int(self.labels[idx])
		return img, label
	

class ImageNetC(Dataset):
    """
    PyTorch Dataset for folder-based ImageNet-C:
    imagenet-c/<corruption>/<severity>/<class>/*.JPEG
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        # Loop through all corruptions and severities
        for corruption in sorted(os.listdir(root_dir)):
            corruption_path = os.path.join(root_dir, corruption)
            if not os.path.isdir(corruption_path):
                continue

            for severity in sorted(os.listdir(corruption_path)):
                severity_path = os.path.join(corruption_path, severity)
                if not os.path.isdir(severity_path):
                    continue

                # Use ImageFolder to get (image_path, label) pairs
                subset = datasets.ImageFolder(severity_path)

                for img_path, label in subset.samples:
                    self.samples.append({
                        'path': img_path,
                        'label': label,
                        'corruption': corruption,
                        'severity': int(severity),
                        'class_to_idx': subset.class_to_idx
                    })

        # Build global class-to-idx mapping
        self.class_to_idx = self.samples[0]['class_to_idx'] if self.samples else {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, sample['label'], sample['corruption'], sample['severity']

class FewShotDataset(Dataset):
	"""
	Same generic image dataset, but this time for each individual sample we give few-shot examples for calibration
	"""
	def __init__(self, data, labels, n_classes, n_supp, n_queries, transform=None):
		"""
		images:   Tensor [N, C, H, W] or list of PILs
		labels:   LongTensor [N]
		n_classes:    number of classes per episode/batch
		k_shot:   support examples per class
		q_queries: query examples per class
		"""
		self.n_classes = n_classes
		self.n_supp = n_supp
		self.n_queries = n_queries

		# Load data
		self.data = data#.numpy()
		if isinstance(self.data[0], np.ndarray):
			self.data = torch.stack([transforms.ToPILImage()(img) for img in self.data])

		self.labels = labels
		
		self.transform = transform

		# Create support and querry structures
		# group indices by class
		self.idx_by_class = defaultdict(list)
		for idx, label in enumerate(self.labels):
			self.idx_by_class[label].append(idx)
		#print(self.idx_by_class)
		self.classes = list(self.idx_by_class.keys())

	# number of episodes
	def __len__(self):
		return 1000
	
	def __getitem__(self, idx):
		# sample n_way classes
		sampled_classes = random.sample(self.classes, self.n_classes)

		# convert global labels to local episode labels(0. .. n_classes-1)
		class2local = {g: i for i, g in enumerate(sampled_classes)}

		supp_idxs = []
		query_idxs = []
		for g in sampled_classes:
			ids = random.sample(self.idx_by_class[g], self.n_supp + self.n_queries)
			supp_idxs.extend(ids[:self.n_supp])
			query_idxs.extend(ids[self.n_supp:])


		supp_imgs = self.data[supp_idxs]
		query_imgs = self.data[query_idxs]
		supp_labels = torch.LongTensor([
			class2local[self.labels[idx]] for idx in supp_idxs
		])
		query_labels = torch.LongTensor([
			class2local[self.labels[idx]] for idx in query_idxs
		])


		if self.transform:
			supp_imgs = torch.stack([self.transform(img) for img in supp_imgs])
			query_imgs = torch.stack([self.transform(img) for img in query_imgs])

		return supp_imgs, supp_labels, query_imgs, query_labels
		



# supp_data [B, n_Cl, C, H, W]
# supp_lb [B, n_Cl]
# query_data [B, n_Cl, C, H, W]
# query_lb [B, n_Cl]



