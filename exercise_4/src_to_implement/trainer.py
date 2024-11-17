import numpy as np
import torch
import torch as t
from sklearn.metrics import f1_score


class Trainer:

	def __init__(self,
				 model,  # Model to be trained.
				 crit,  # Loss function
				 optim=None,  # Optimizer
				 train_dl=None,  # Training data set
				 val_test_dl=None,  # Validation (or test) data set
				 cuda=True,  # Whether to use the GPU
				 early_stopping_patience=-1):  # The patience for early stopping
		self._model = model
		self._crit = crit
		self._optim = optim
		self._train_dl = train_dl
		self._val_test_dl = val_test_dl
		self._cuda = cuda

		self._early_stopping_patience = early_stopping_patience

		if cuda:
			self._model = model.cuda()
			self._crit = crit.cuda()
		
		self.best_f = 0

	def save_checkpoint(self, epoch):
		t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

	def restore_checkpoint(self, epoch_n):
		ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
		self._model.load_state_dict(ckp['state_dict'])

	def save_onnx(self, fn):
		m = self._model.cpu()
		m.eval()
		x = t.randn(1, 3, 300, 300, requires_grad=True)
		y = self._model(x)
		t.onnx.export(m,  # model being run
					  x,  # model input (or a tuple for multiple inputs)
					  fn,  # where to save the model (can be a file or file-like object)
					  export_params=True,  # store the trained parameter weights inside the model file
					  opset_version=10,  # the ONNX version to export the model to
					  do_constant_folding=True,  # whether to execute constant folding for optimization
					  input_names=['input'],  # the model's input names
					  output_names=['output'],  # the model's output names
					  dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
									'output': {0: 'batch_size'}})

	def train_step(self, x, y):
		# perform following steps:
		# -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called.
		# This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
		# -propagate through the network
		# -calculate the loss
		# -compute gradient by backward propagation
		# -update weights
		# -return the loss
		# TODO
		self._optim.zero_grad()
		output = self._model.forward(x)
		loss = self._crit(output, y)
		loss.backward()
		self._optim.step()
		return loss

	def val_test_step(self, x, y):

		# predict
		# propagate through the network and calculate the loss and predictions
		# return the loss and the predictions
		# TODO
		output = self._model.forward(x)
		loss = self._crit(output, y)
		# print("prediction: ", torch.round(output))
		# print("truth: ", y)
		# print("f1 score: ", score)
		return loss, output

	def train_epoch(self):
		# set training mode
		# iterate through the training set
		# transfer the batch to "cuda()" -> the gpu if a gpu is given
		# perform a training step
		# calculate the average loss for the epoch and return it
		# TODO
		self._model.train()
		loss = []
		for x, y in self._train_dl:
			if self._cuda:
				x = x.cuda()
				y = y.cuda()
			loss.append(self.train_step(x, y))
		return torch.mean(torch.tensor(loss))

	def val_test(self):
		# set eval mode. Some layers have different behaviors during training and testing
		# (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
		# disable gradient computation. Since you don't need to update the weights during testing,
		# gradients aren't required anymore.
		# iterate through the validation set
		# transfer the batch to the gpu if given
		# perform a validation step
		# save the predictions and the labels for each batch
		# calculate the average loss and average metrics of your choice.
		# You might want to calculate these metrics in designated functions
		# return the loss and print the calculated metrics
		# TODO
		self._model.eval()
		# for p in self._model.parameters():
		#     p.require_grads = False

		truth = None
		prediction = None
		with torch.no_grad():
			loss_all = []
			for x, y in self._val_test_dl:
				if self._cuda:
					x = x.cuda()
					y = y.cuda()
				loss, pred = self.val_test_step(x, y)
				loss_all.append(loss)

				if truth is None:
					truth = y.cpu().detach().numpy()
				else:
					truth = np.concatenate((truth, y.cpu().detach().numpy()), axis=0)
				if prediction is None:
					prediction = torch.round(pred).cpu().detach().numpy()
				else:
					prediction = np.concatenate((prediction, torch.round(pred).cpu().detach().numpy()), axis = 0)
			score = f1_score(truth, prediction, average="micro")
			if self.best_f < score:
				self.best_f = score
			print(score)
		return torch.mean(torch.tensor(loss_all))

	def fit(self, epochs=-1):
		assert self._early_stopping_patience > 0 or epochs > 0
		# create a list for the train and validation losses, and create a counter for the epoch
		# TODO
		train_loss = []
		val_loss = []
		counter = 0
		f = 0

		while counter < epochs:
			print(counter)
			# stop by epoch number
			# train for a epoch and then calculate the loss and metrics on the validation set
			# append the losses to the respective lists
			# use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
			# check whether early stopping should be performed using the early stopping criterion and stop if so
			# return the losses for both training and validation
			# TODO
			t_l = self.train_epoch().detach().numpy()
			train_loss.append(t_l)
			v_l = self.val_test().detach().numpy()
			val_loss.append(v_l)
			# if counter % 5 == 0 and counter >= 20:
			# 	self.save_checkpoint(counter)
			if counter >= 5 and f < self.best_f:
				f = self.best_f
				self.save_checkpoint(counter)
			# print(train_loss)
			# print(val_loss)
			if self.early_stop(val_loss):
				break
			counter += 1
		return train_loss, val_loss

	def early_stop(self, val_loss):
		loss_count = len(val_loss)
		if 0 < self._early_stopping_patience * 2 < loss_count:
			prev_loss = cur_loss = 0
			for l in range(loss_count - (self._early_stopping_patience * 2), loss_count):
				if l < loss_count - self._early_stopping_patience:
					prev_loss += val_loss[l]
				else:
					cur_loss += val_loss[l]
			prev_loss /= self._early_stopping_patience
			cur_loss /= self._early_stopping_patience
			# print(cur_loss, prev_loss)
			# if cur_loss - prev_loss > .1:
			# 	return True
		return False
