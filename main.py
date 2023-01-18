### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs
from ImageUtils import visualize


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="train, test or predict")
# parser.add_argument("--mode", type=str, default="predict", help="train, test or predict")
parser.add_argument("--data_dir", type=str, default="cifar-10-batches-py", help="path to the data")
parser.add_argument("--test_data_dir", type=str, default="cifar-10-batches-py-private", help="path to the data")
parser.add_argument("--save_dir", type=str, default="result_dir", help="path to save the results")
args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(model_configs, training_configs)

	if args.mode == 'train':
		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		# x_train, y_train, x_test, y_test = load_data('cifar-10-batches-py')
		x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)

		model.train(x_train_new, y_train_new, training_configs, x_valid, y_valid)
		# Training with validation data and then go for test
		# model.train(x_train, y_train, training_configs)
		# model.evaluate(x_test, y_test, list(range(10, 210, 10)))
		# model.evaluate(x_valid, y_valid, list(range(139, 162)) + [170, 180])
		# model.evaluate(x_valid, y_valid, list(range(10, 210, 10)))
		model.evaluate(x_valid, y_valid, [160])

		# Training with validation data and then go for test
		# model.train(x_train, y_train, training_configs)


	elif args.mode == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)
		# _, _, x_test, y_test = load_data('cifar-10-batches-py')
		# model.evaluate(x_test, y_test, list(range(10, 210, 10)))
		model.evaluate(x_test, y_test, [160])

	elif args.mode == 'predict':
		# Loading private testing dataset
		x_predict = load_testing_images(args.test_data_dir)
		# visualizing the first testing image to check your image shape
		# visualize(x_test[0], 'test.png')
		# Predicting and storing results on private testing dataset 
		predictions = model.predict_prob(x_predict, [160])
		# np.save(args.result_dir, predictions)
		with open(args.save_dir + '/predictions.npy', 'wb') as f:
			np.save(f, np.array(predictions))
		

### END CODE HERE

