import math
import numpy as np

class Evaluation:

	def compute_rank1(self, Y, y):
		classes = np.unique(sorted(y))
		count_all = 0
		count_correct = 0
		for cla1 in classes:
			idx1 = y==cla1
			if (list(idx1).count(True)) <= 1:
				continue
			# Compute only for cases where there is more than one sample:
			Y1 = Y[idx1==True, :]
			Y1[Y1==0] = math.inf
			for y1 in Y1:
				s = np.argsort(y1)
				smin = s[0]
				imin = idx1[smin]
				count_all += 1
				if imin:
					count_correct += 1
		return count_correct/count_all*100


	def compute_rank1_train(self, X, y_train, y_test):
		"""
		Computes rank1 on test set with train set used as our database
		:param X: np.array of shape (len(y_test), len(y_train)), stores the distance between each test and train instance
		:param y_train: list of classes (IDs) of train images
		:param y_test: list of classes (IDs) of test images
		:return:
		"""
		test_classes = np.unique(sorted(y_test))
		count_all = 0
		count_correct = 0
		for cla1 in test_classes:
			idx_train = y_train == cla1
			if (list(idx_train).count(True)) == 0:
				continue
			# Compute only for cases where there is person enrolled in database
			idx_test = y_test == cla1
			X1 = X[idx_test == True, :]
			for x in X1:
				s = np.argsort(x)
				count_all += 1
				if idx_train[s[0]]:
					count_correct += 1
		return count_correct / count_all * 100

	def compute_ranks(self, X, y_train, y_test):
		train_classes = np.unique(sorted(y_train))
		rank_scores = []

		s = np.argsort(X, axis=1)

		# Go through all ranks
		match = np.repeat(False, X.shape[0])
		for i in range(X.shape[1]):
			preds = s[:, i]
			# Check if we hit the class
			true_preds = np.array([y_train[pred] == true_class for (pred, true_class) in zip(preds, y_test)])
			match = np.logical_or(match, true_preds)

			rank_scores.append(np.sum(match) / X.shape[0])

		return rank_scores

	def compute_rank1_probability(self, Y, y_true):
		"""
		Computes rank1 based on class probability predictions
		:param Y: np.array of shape (test_size, n_classes) class probability predictions
		:param y_true: list of true classes (IDs)
		:return:
		"""
		predicted_class = np.argmax(Y, axis=1)

		return 100* np.mean(predicted_class == np.array(y_true))




	# Add your own metrics here, such as rank5, (all ranks), CMC plot, ROC, ...

		# def compute_rank5(self, Y, y):
	# 	# First loop over classes in order to select the closest for each class.
	# 	classes = np.unique(sorted(y))
		
	# 	sentinel = 0
	# 	for cla1 in classes:
	# 		idx1 = y==cla1
	# 		if (list(idx1).count(True)) <= 1:
	# 			continue
	# 		Y1 = Y[idx1==True, :]

	# 		for cla2 in classes:
	# 			# Select the closest that is higher than zero:
	# 			idx2 = y==cla2
	# 			if (list(idx2).count(True)) <= 1:
	# 				continue
	# 			Y2 = Y1[:, idx1==True]
	# 			Y2[Y2==0] = math.inf
	# 			min_val = np.min(np.array(Y2))
	# 			# ...