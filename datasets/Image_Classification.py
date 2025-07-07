import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict
import random
import pickle

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
	Generic image classification dataset class.
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
		img = transforms.ToPILImage()(img)
		if self.transform:
			img = self.transform(img)
		label = int(self.labels[idx])
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
	A PyTorch Dataset for ImageNet‐C.  
	We expect IMAGENET_C_ROOT to contain directories like `gaussian_blur/1/…`, `gaussian_blur/2/…`, etc.
	We walk all corruption types and severities, collect all image paths, then map filename → label using the dictionary above.
	"""
	def __init__(self, root: str, filename_to_label: dict, transform=None):
		"""
		root: path to “imagenet-c” folder (e.g. "/…/imagenet/imagenet-c")
		filename_to_label: dict mapping "ILSVRC2012_val_00000001.JPEG" -> integer class (0–999)
		transform: torchvision transforms to apply to each corrupted image
		"""
		super().__init__()
		self.root = Path(root)
		self.filename_to_label = filename_to_label
		self.transform = transform

		self.samples = []  # will hold tuples: (image_path, label, corruption_name, severity_int)

		# Walk through every “corruption_name” directory under root:
		for corruption_dir in sorted(self.root.iterdir()):
			if not corruption_dir.is_dir():
				continue
			corruption_name = corruption_dir.name  # e.g. "gaussian_blur"
			# Inside each corruption directory, there should be five subfolders: "1", "2", … "5"
			for severity_folder in sorted(corruption_dir.iterdir()):
				if not severity_folder.is_dir():
					continue
				try:
					severity = int(severity_folder.name)  # 1 through 5
				except ValueError:
					# skip if folder isn’t named “1”–“5”
					continue

				# Now gather all JPEGs in this severity folder:
				for img_path in severity_folder.glob("*.JPEG"):
					fname = img_path.name  # e.g. "ILSVRC2012_val_00000001.JPEG"
					if fname not in self.filename_to_label:
						# If a filename isn’t found in the clean‐val mapping, skip it
						continue
					label = self.filename_to_label[fname]
					self.samples.append((str(img_path), label, corruption_name, severity))

		print(f"Collected {len(self.samples)} ImageNet‐C samples across all corruptions/severities.")

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		path, label, corruption_name, severity = self.samples[idx]
		img = Image.open(path).convert("RGB")
		if self.transform is not None:
			img = self.transform(img)
		# We return a tuple of (tensor_img, label, corruption_name, severity)
		return img, label, corruption_name, severity
	

class FewShotDataset(nn.Module):
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
		self.labels = labels.numpy()
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

		supp_idxs = []
		query_idxs = []
		#print(f"Sampled classes: {sampled_classes}")
		for cl in sampled_classes:
			#print(len(self.idx_by_class[cl]), "samples for class", cl)
			#print(f"Sampling {self.n_supp + self.n_queries} indices for class {cl}")
			idxs = random.sample(self.idx_by_class[cl], self.n_supp + self.n_queries)
			#print(f"Class {cl} indices: {idxs}")
			supp_idxs += idxs[:self.n_supp]
			query_idxs += idxs[self.n_supp:]

		supp_imgs = self.data[supp_idxs]
		query_imgs = self.data[query_idxs]
		supp_labels = torch.LongTensor(self.labels[supp_idxs])
		query_labels = torch.LongTensor(self.labels[query_idxs])

		#print(f"Supp imgs shape: {supp_imgs.shape}, Supp labels shape: {supp_labels.shape}")
		#print(f"Query imgs shape: {query_imgs.shape}, Query labels shape: {query_labels.shape}")
		return supp_imgs, supp_labels, query_imgs, query_labels
		



# supp_data [B, n_Cl, C, H, W]
# supp_lb [B, n_Cl]
# query_data [B, n_Cl, C, H, W]
# query_lb [B, n_Cl]



