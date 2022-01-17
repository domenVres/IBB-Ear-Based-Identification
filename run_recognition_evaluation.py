import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist 
from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation

import re
from tqdm.auto import tqdm

import feature_extractors.pix2pix.extractor as p2p_ext
from feature_extractors.lbp.extractor import LBP

import matplotlib.pyplot as plt

from data.AWEDataset import AWETestSet
from torchvision import transforms
import torch.nn as nn
from feature_extractors.your_super_extractor.CNN_detector import CNNEarDetector
from preprocessing.preprocess import HistogramEqualization, ImageSharpening, EdgeEnhancement

import warnings
warnings.filterwarnings('ignore')


NUM_CLASSES = 100
INPUT_SIZE = 224

data_transforms = {
    'none': transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'histogram': transforms.Compose([
        HistogramEqualization(),
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'edge': transforms.Compose([
        EdgeEnhancement(),
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'sharpen': transforms.Compose([
        ImageSharpening(),
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


class EvaluateAll:

    def __init__(self, config_path='config_recognition.json'):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open(config_path) as config_file:
            config = json.load(config_file)

        self.train_images_path = config['train_images_path']
        self.test_images_path = config['test_images_path']
        self.annotations_path = config['annotations_path']

        self.eval = Evaluation()

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]

    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d


    def evaluate_CNN(self, model, preprocessing, all_ranks=False):
        """
        Function that calculates rank1 (accuracy) of given CNN model and all ranks if specified
        :param model: either "resnet" or "alexnet"
        :param preprocessing: either None, "histogram", "edge" or "sharpen"
        :param y_true: list of true class values for test set
        :param all_ranks:
        :return:
        """
        preprocess_path = ""
        if preprocessing == "histogram":
            preprocess_path = "_equalization"
        elif preprocessing == "edge":
            preprocess_path = "_edge"
        elif preprocessing == "sharpen":
            preprocess_path = "_sharpening"
        model_name = f"trained_{model}{preprocess_path}"

        Y_probs, y_true = get_CNN_predictions(model, model_name, preprocessing)

        rank1 = self.eval.compute_rank1_probability(Y_probs, y_true)

        ranks = None
        if all_ranks:
            ranks = self.eval.compute_ranks(Y_probs, list(range(1, 101)), y_true)

        return rank1, ranks


    def run_evaluation(self, all_ranks=False, preprocessing=None):

        test_im_list = sorted(glob.glob(self.test_images_path + '/*.png', recursive=True))
        train_im_list = sorted(glob.glob(self.train_images_path + '/*.png', recursive=True))
        iou_arr = []
        preprocess = Preprocess()

        cla_d = self.get_annotations(self.annotations_path)
        
        # Change the following extractors, modify and add your own

        # Pixel-wise comparison:
        pix2pix = p2p_ext.Pix2Pix()

        # LBP feature extraction
        lbp = LBP(radius=5, num_points=32)
        
        lbp_features_train = []
        plain_features_train = []
        lbp_features_test = []
        plain_features_test = []
        y_train = []
        y_test = []

        for im_name in tqdm(train_im_list):
            
            # Read an image
            img = cv2.imread(im_name)

            # Correct the path for windows and get the correct class
            a_name = re.sub("\\\\", "/", im_name)
            y_train.append(cla_d['/'.join(a_name.split('/')[-2:]).lstrip("original_")])

            # Apply some preprocessing
            if preprocessing == "histogram":
                img = preprocess.histogram_equalization_rgb(img)  # This one makes VJ worse
            elif preprocessing == "edge":
                img = preprocess.edge_enhancement(img)
            elif preprocessing == "sharpen":
                img = preprocess.image_sharpening(img)
            
            # Run the feature extractors            
            plain_features = pix2pix.extract(img)
            plain_features_train.append(plain_features)
            lbp_features = lbp.extract(img)
            lbp_features_train.append(lbp_features)

        for im_name in tqdm(test_im_list):
            # Read an image
            img = cv2.imread(im_name)

            # Correct the path for windows and get the correct class
            a_name = re.sub("\\\\", "/", im_name)
            y_test.append(cla_d['/'.join(a_name.split('/')[-2:])])

            # Apply some preprocessing
            if preprocessing == "histogram":
                img = preprocess.histogram_equalization_rgb(img)  # This one makes VJ worse
            elif preprocessing == "edge":
                img = preprocess.edge_enhancement(img)
            elif preprocessing == "sharpen":
                img = preprocess.image_sharpening(img)

            # Run the feature extractors
            plain_features = pix2pix.extract(img)
            plain_features_test.append(plain_features)
            lbp_features = lbp.extract(img)
            lbp_features_test.append(lbp_features)

        X_plain = cdist(plain_features_test, plain_features_train, 'jensenshannon')

        r1 = self.eval.compute_rank1_train(X_plain, y_train, y_test)
        print('Pix2Pix Rank-1[%]', r1)

        X_lbp = cdist(lbp_features_test, lbp_features_train, 'jensenshannon')

        r1 = self.eval.compute_rank1_train(X_lbp, y_train, y_test)
        print('LBP Rank-1[%]', r1)
        print()

        resnet_ranks = self.evaluate_CNN("resnet", preprocessing, all_ranks)
        print('ResNet50 Rank-1[%]', resnet_ranks[0])
        print()

        alexnet_ranks = self.evaluate_CNN("alexnet", preprocessing, all_ranks)
        print('AlexNet Rank-1[%]', alexnet_ranks[0])
        print()

        if all_ranks:
            # Compute the all ranks and plot the data
            plain_ranks = self.eval.compute_ranks(X_plain, y_train, y_test)
            lbp_ranks = self.eval.compute_ranks(X_lbp, y_train, y_test)
            rank = list(range(1, len(plain_ranks)+1))
            cnn_rank = list(range(1, len(resnet_ranks[1])+1))

            plt.plot(rank, plain_ranks, label="Pixel to pixel")
            plt.plot(rank, lbp_ranks, label="LBP")
            plt.plot(cnn_rank, resnet_ranks[1], label="ResNet50")
            plt.plot(cnn_rank, alexnet_ranks[1], label="AlexNet")
            plt.xlabel("Rank")
            plt.ylabel("Ratio of correct cases")
            plt.title("Graph of all ranks")
            plt.legend()
            plt.savefig("Ranks.png")
            plt.show()


def get_CNN_predictions(model, model_name, preprocess):
    """
    Method that returns class probability predictions obtained with CNN for each test set instance
    :param model_path:
    :param preprocess:
    :return: (preds, y) preds: np.array of shape (n_instances, n_classes), matrix of probabilities
                        y: list of length n_instances, true class values
    """
    # Load model
    model = CNNEarDetector(model=model, num_classes=NUM_CLASSES, extract_features=False)
    model.load(os.getcwd(), model_name)
    model.set_eval()

    # Load test data
    if preprocess is None:
        preprocess = "none"
    test_data = AWETestSet(os.path.join(os.getcwd(), "data/perfectly_detected_ears"), data_transforms[preprocess])

    # Placeholders for results
    preds = []
    y = []
    # So we can transform outputs to probabilities
    get_probs = nn.Softmax()

    # Go through test_data and make predictions
    for i in range(len(test_data)):
        image, label = test_data[i]
        y.append(int(label))
        image = image[None, :].to('cuda')
        pred = get_probs(model(image))
        preds.append(pred.squeeze(0).to('cpu').detach().numpy())

    preds = np.array(preds)

    return preds, y

if __name__ == '__main__':
    ev = EvaluateAll()

    print("Without preprocessing:")
    ev.run_evaluation(all_ranks=True)
    print("Histogram equalization:")
    ev.run_evaluation(preprocessing="histogram")
    print("Edge enhancement:")
    ev.run_evaluation(preprocessing="edge")
    print("Image sharpening:")
    ev.run_evaluation(preprocessing="sharpen")

    """ev_mask_cnn = EvaluateAll(config_path="config_recognition_mask-r-cnn.json")
    print("Without preprocessing:")
    ev_mask_cnn.run_evaluation()
    print("Histogram equalization:")
    ev_mask_cnn.run_evaluation(preprocessing="histogram")
    print("Edge enhancement:")
    ev_mask_cnn.run_evaluation(preprocessing="edge")
    print("Image sharpening:")
    ev_mask_cnn.run_evaluation(preprocessing="sharpen")"""
