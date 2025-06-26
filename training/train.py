import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

#from training.train import KoleoLoss#, SinkhornCentering

from utils.losses import KoLeoLoss

def format_time(seconds: float) -> str:
	seconds = int(seconds)
	mins, sec = divmod(seconds, 60)
	hrs, mins = divmod(mins, 60)
	return f"{hrs}h{mins:02d}m{sec:02d}s"

# The stand-alone training function for one baseline model
def train_model(
	name: str,
	model: nn.Module,
	optimizer: optim.Optimizer,
	train_loader: DataLoader,
	test_loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	num_epochs: int
) -> None:
	"""
	Train and validate a single baseline model over `num_epochs` epochs.
	Prints per‐epoch train/val metrics, epoch time, and ETA. 
	"""
	model.to(device)
	best_val_acc = 0.0

	ce_fn = criterion#nn.CrossEntropyLoss()
	koleo_fn = KoLeoLoss()
	#sinkhorn_fn = SinkhornCentering(iters=3, eps=0.04) 

	print(f"\n=== Training baseline: {name} ===")
	model_start_time = time.time()

	for epoch in range(1, num_epochs + 1):
		epoch_start = time.time()

		# Train for one epoch
		model.train()
		running_loss = 0.0
		correct = 0
		total = 0
		for inputs, targets in train_loader:
			inputs, targets = inputs.to(device), targets.to(device)
			optimizer.zero_grad()
			feat, outputs = model(inputs)
			#print(outputs.shape, targets.shape)
			#cross entropy loss
			ce_loss = ce_fn(outputs, targets)
			# sinkhorn lossba
			#Q = sinkhorn_fn(outputs)
			# KoLeo loss
			k_loss = koleo_fn(outputs)
			# Combine losses
			loss = ce_loss + 0.1 * k_loss

			loss.backward()
			optimizer.step()

			running_loss += loss.item() * inputs.size(0)
			_, preds = outputs.max(1)
			correct += (preds == targets).sum().item()
			total += targets.size(0)

		train_loss = running_loss / total
		train_acc = correct / total

		# Validate on test set
		model.eval()
		val_running_loss = 0.0
		val_correct = 0
		val_total = 0
		with torch.no_grad():
			for inputs, targets in test_loader:
				inputs, targets = inputs.to(device), targets.to(device)
				feat, outputs = model(inputs)
				loss = criterion(outputs, targets)
				val_running_loss += loss.item() * inputs.size(0)
				_, preds = outputs.max(1)
				val_correct += (preds == targets).sum().item()
				val_total += targets.size(0)

		val_loss = val_running_loss / val_total
		val_acc = val_correct / val_total

		# Update best validation accuracy
		if val_acc > best_val_acc:
			best_val_acc = val_acc

		epoch_end = time.time()
		epoch_time = epoch_end - epoch_start
		remaining_time = epoch_time * (num_epochs - epoch)

		print(
			f"[{name}][Epoch {epoch}/{num_epochs}] "
			f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}  "
			f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}  "
			f"(best={best_val_acc:.4f})  "
			f"epoch_time={format_time(epoch_time)}, "
			f"ETA={format_time(remaining_time)}"
		)

	total_time = time.time() - model_start_time
	print(f"--- Finished {name} (total_time={format_time(total_time)}) ---\n")
