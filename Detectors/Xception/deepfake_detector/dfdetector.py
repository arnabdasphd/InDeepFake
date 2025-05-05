import argparse
import copy
import os
import shutil
import test
import time
import zipfile
import timm
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import metrics
import matplotlib.pyplot as plt
import cv2
import datasets
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import train
import utils
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from albumentations import (
    Compose, FancyPCA, GaussianBlur, GaussNoise, HorizontalFlip,
    HueSaturationValue, ImageCompression, OneOf, PadIfNeeded,
    RandomBrightnessContrast, Resize, ShiftScaleRotate, ToGray)
from pretrained_mods import xception
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from facedetector.retinaface import df_retinaface
from pretrained_mods import efficientnetb1lstm
from pretrained_mods import mesonet
from pretrained_mods import resnetlstm
from utils import vidtimit_setup_real_videos



parser = argparse.ArgumentParser(
    description='Start deepfake detection.')
parser.add_argument('--detect_single', action='store_true', help='Detect deepfake in a single video or image.')
parser.add_argument('--benchmark', default=False, type=bool,
                    help='Choose for benchmarking.')
parser.add_argument('--train', default=False, type=bool,
                    help='Choose for training.')
parser.add_argument('--path_to_vid', default=None,
                    type=str, help='Choose video path.')
parser.add_argument('--path_to_img', default=None,
                    type=str, help='Choose image path.')
parser.add_argument('--detection_method', default="xception_uadfv",
                    type=str, help='Choose detection method.')
parser.add_argument('--data_path', default=None, type=str,
                    help='Specify path to dataset.')
parser.add_argument('--dataset', default="celebdf", type=str,
                    help='Specify the name of the dataset.')
parser.add_argument('--cmd', default="True", type=str,
                    help='True if executed via command line.')
parser.add_argument('--model_type', default="xception",
                    type=str, help='Choose detection model type for training.')
parser.add_argument('--epochs', default=1,
                    type=int, help='Choose number of training epochs.')
parser.add_argument('--batch_size', default=32,
                    type=int, help='Choose the minibatch size.')
parser.add_argument('--lr', default=0.0001,
                    type=int, help='Choose the minibatch size.')
parser.add_argument('--folds', default=1,
                    type=int, help='Choose validation folds.')
parser.add_argument('--augs', default="weak",
                    type=str, help='Choose augmentation strength.')
parser.add_argument('--fulltrain', default=False,
                    type=bool, help='Choose whether to train with the full dataset and no validation set.')
parser.add_argument('--facecrops_available', default=False,
                    type=bool, help='Choose whether videos are already preprocessed.')
parser.add_argument('--face_margin', default=0.3,
                    type=float, help='Choose the face margin.')
parser.add_argument('--seed', default=24,
                    type=int, help='Choose the random seed.')
parser.add_argument('--save_path', default=None,
                    type=str, help='Choose the path where face crops shall be saved.')                       
parser.add_argument('--directory_path', type=str, help='Path to the directory containing videos.')                                                             
parser.add_argument('--real_videos_path', type=str, help='Path to the directory containing real videos.')
parser.add_argument('--fake_videos_path', type=str, help='Path to the directory containing fake videos.')                                      


class DFDetector():
    """
    The Deepfake Detector. 
    It can detect on a single video, 
    benchmark several methods on benchmark datasets
    and train detectors on several datasets.
    """

    def __init__(self):
        pass

    @classmethod
    def detect_single(cls, video_path=None, image_path=None, label=None, method="xception_uadfv", cmd=False):
        """Perform deepfake detection on a single video with a chosen method."""
        # prepare the method of choice
        sequence_model = False
        if method == "xception_celebdf":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "Xception_CELEB-DF"
        
        if video_path:
            if method not in ["dfdcrank90_uadfv", "dfdcrank90_celebdf", "dfdcrank90_dfdc", "dfdcrank90_dftimit_lq", "dfdcrank90_dftimit_hq", "six_method_ensemble_uadfv", "six_method_ensemble_celebdf", "six_method_ensemble_dftimit_lq", "six_method_ensemble_dftimit_hq", "six_method_ensemble_dfdc"]:
                data = [[1, video_path]]
                df = pd.DataFrame(data, columns=['label', 'video'])
                try:
                    loss = test.inference(
                        model, df, img_size, normalization, dataset=None, method=method, face_margin=0.3, sequence_model=sequence_model, num_frames=20, single=True, cmd=cmd)

                    if round(loss) == 1:
                        result = "Deepfake detected."
                        print("Deepfake detected.")
                        return round(loss)
                    else:
                        result = "This is a real video."
                        print("This is a real video.")
                        return round(loss)
                except IndexError:
                    print(f"No faces detected in video {video_path}.")
                    return -1 

    @classmethod
    def benchmark(cls, dataset=None, data_path=None, method="xception_celebdf", seed=24):
        """Benchmark deepfake detection methods against popular deepfake datasets.
           The methods are already pretrained on the datasets. 
           Methods get benchmarked against a test set that is distinct from the training data.
        # Arguments:
            dataset: The dataset that the method is tested against.
            data_path: The path to the test videos.
            method: The deepfake detection method that is used.
        # Implementation: Christopher Otto
        """
        # seed numpy and pytorch for reproducibility
        reproducibility_seed(seed)
        if method not in ['xception_celebdf']:
            raise ValueError("Method is not available for benchmarking.")
        else:
            # method exists
            cls.dataset = dataset
            cls.data_path = data_path
            cls.method = method
            if method in []:
                face_margin = 0.0
            else:
                face_margin = 0.3
        if cls.dataset == 'celebdf':
            num_frames = 20
            setup_celebdf_benchmark(cls.data_path, cls.method)
        # get test labels for metric evaluation
        df = label_data(dataset_path=cls.data_path,
                        dataset=cls.dataset, test_data=True)
        # prepare the method of choice
        if cls.method == 'xception_celebdf':
            model, img_size, normalization = prepare_method(
                method=cls.method, dataset=cls.dataset, mode='test')
            return [auc, ap, loss, acc]

        print(f"Detecting deepfakes with \033[1m{cls.method}\033[0m ...")

    @classmethod
    def train_method(cls, dataset=None, data_path=None, method="xception", img_save_path=None, epochs=1, batch_size=32,
                     lr=0.001, folds=1, augmentation_strength='weak', fulltrain=False, faces_available=False, face_margin=0, seed=24):
        """Train a deepfake detection method on a dataset."""
        if img_save_path is None:
            raise ValueError(
                "Need a path to save extracted images for training.")
        cls.dataset = dataset
        print(f"Training on {cls.dataset} dataset.")
        cls.data_path = data_path
        cls.method = method
        cls.epochs = epochs
        cls.batch_size = batch_size
        cls.lr = lr
        cls.augmentations = augmentation_strength
        # no k-fold cross val if folds == 1
        cls.folds = folds
        # whether to train on the entire training data (without val sets)
        cls.fulltrain = fulltrain
        cls.faces_available = faces_available
        cls.face_margin = face_margin
        print(f"Training on {cls.dataset} dataset with {cls.method}.")
        # seed numpy and pytorch for reproducibility
        reproducibility_seed(seed)
        #folder_count = 35
        _, img_size, normalization = prepare_method(
            cls.method, dataset=cls.dataset, mode='train')
        # # get video train data and labels
        df = label_data(dataset_path=cls.data_path,
                        dataset=cls.dataset, test_data=False, fulltrain=cls.fulltrain)
        # detect and extract faces if they are not available already
        if not cls.faces_available:
            if cls.dataset == 'celebdf':
                addon_path = '/facecrops/'
                # check if all folders are available
                if not os.path.exists(img_save_path + '/Celeb-real/'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"Celeb-real\" folder is missing.")
                if not os.path.exists(img_save_path + '/Celeb-synthesis/'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"Celeb-synthesis\" folder is missing.")
                if not os.path.exists(img_save_path + '/YouTube-real/'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"YouTube-real\" folder is missing.")
                if not os.path.exists(img_save_path + '/List_of_testing_videos.txt'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"List_of_testing_videos.txt\" file is missing.")
                if not os.path.exists(img_save_path + '/facecrops/'):
                    # create directory in save path for face crops
                    os.mkdir(img_save_path + addon_path)
                    os.mkdir(img_save_path + '/facecrops/real/')
                    os.mkdir(img_save_path + '/facecrops/fake/')
                else:
                    # delete create again if it already exists with old files
                    shutil.rmtree(img_save_path + '/facecrops/')
                    os.mkdir(img_save_path + addon_path)
                    os.mkdir(img_save_path + '/facecrops/real/')
                    os.mkdir(img_save_path + '/facecrops/fake/')
            

            if cls.dataset == 'dfdc':
                num_frames = 5
            else:
                num_frames = 20
            print(
                f"Detect and save {num_frames} faces from each video for training.")
            if cls.face_margin > 0.0:
                print(
                    f"Apply {cls.face_margin*100}% margin to each side of the face crop.")
            else:
                print("Apply no margin to the face crop.")
            # load retinaface face detector
            net, cfg = df_retinaface.load_face_detector()
            for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
                video = row.loc['video']
                label = row.loc['label']
                vid = os.path.join(video)
                if cls.dataset == 'celebdf':
                    vid_name = row.loc['video_name']
                    if label == 1:
                        video = vid_name
                        save_dir = os.path.join(
                            img_save_path + '/facecrops/fake/')
                    else:
                        video = vid_name
                        save_dir = os.path.join(
                            img_save_path + '/facecrops/real/')
                

                # detect faces, add margin, crop, upsample to same size, save to images
                faces = df_retinaface.detect_faces(
                    net, vid, cfg, num_frames=num_frames)
                # save frames to directory
                vid_frames = df_retinaface.extract_frames(
                    faces, video, save_to=save_dir, face_margin=cls.face_margin, num_frames=num_frames, test=False)

        # put all face images in dataframe
        df_faces = label_data(dataset_path=cls.data_path,
                              dataset=cls.dataset, method=cls.method, face_crops=True, test_data=False, fulltrain=cls.fulltrain)
        # choose augmentation strength
        augs = df_augmentations(img_size, strength=cls.augmentations)
        # start method training

        model, average_auc, average_ap, average_acc, average_loss = train.train(dataset=cls.dataset, data=df_faces,
                                                                                method=cls.method, img_size=img_size, normalization=normalization, augmentations=augs,
                                                                                folds=cls.folds, epochs=cls.epochs, batch_size=cls.batch_size, lr=cls.lr, fulltrain=cls.fulltrain
                                                                                )
        return model, average_auc, average_ap, average_acc, average_loss


def prepare_method(method, dataset, mode='train'):
    """Prepares the method that will be used for training or benchmarking."""
    if method == 'xception' or method == 'xception_uadfv' or method == 'xception_celebdf' or method == 'xception_dftimit_hq' or method == 'xception_dftimit_lq' or method == 'xception_dfdc':
        img_size = 299
        normalization = 'xception'
        if mode == 'test':
            model = xception.imagenet_pretrained_xception()
            # load the xception model that was pretrained on the respective datasets training data
            if method == 'xception_uadfv' or method == 'xception_celebdf' or method == 'xception_dftimit_hq' or method == 'xception_dftimit_lq' or method == 'xception_dfdc':
                model_params = torch.load(
                    os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                print(os.getcwd(
                ) + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params)
            return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    else:
        raise ValueError(
            f"{method} is not available. Please use one of the available methods.")
    

def label_data(dataset_path=None, dataset='uadfv', method='xception', face_crops=False, test_data=False, fulltrain=False):
    """
    Label the data.
    # Arguments:
        dataset_path: path to data
        test_data: binary choice that indicates whether data is for testing or not.
    # Implementation: Christopher Otto
    """
    # structure data from folder in data frame for loading
    if dataset_path is None:
        raise ValueError("Please specify a dataset path.")
    if not test_data:
        if dataset == 'celebdf':
            # prepare celebdf training data by
            # reading in the testing data first
            df_test = pd.read_csv(
                dataset_path + '/List_of_testing_videos.txt', sep=" ", header=None)
            df_test.columns = ["label", "video"]
            # switch labels so that fake label is 1
            df_test['label'] = df_test['label'].apply(switch_one_zero)
            df_test['video'] = dataset_path + '/' + df_test['video']
            # structure data from folder in data frame for loading
            if not face_crops:
                video_path_real = os.path.join(dataset_path + "/Celeb-real/")
                video_path_fake = os.path.join(
                    dataset_path + "/Celeb-synthesis/")
                real_list = []
                for _, _, videos in os.walk(video_path_real):
                    for video in tqdm(videos):
                        # label 0 for real image
                        real_list.append({'label': 0, 'video': video})

                fake_list = []
                for _, _, videos in os.walk(video_path_fake):
                    for video in tqdm(videos):
                        # label 1 for deepfake image
                        fake_list.append({'label': 1, 'video': video})

                # put data into dataframe
                df_real = pd.DataFrame(data=real_list)
                df_fake = pd.DataFrame(data=fake_list)
                # add real and fake path to video file name
                df_real['video_name'] = df_real['video']
                df_fake['video_name'] = df_fake['video']
                df_real['video'] = video_path_real + df_real['video']
                df_fake['video'] = video_path_fake + df_fake['video']
                # put testing vids in list
                testing_vids = list(df_test['video'])
                # remove testing videos from training videos
                df_real = df_real[~df_real['video'].isin(testing_vids)]
                df_fake = df_fake[~df_fake['video'].isin(testing_vids)]
                # undersampling strategy to ensure class balance of 50/50
                df_fake_sample = df_fake.sample(
                    n=len(df_real), random_state=24).reset_index(drop=True)
                # concatenate both dataframes to get full training data (964 training videos with 50/50 class balance)
                df = pd.concat([df_real, df_fake_sample], ignore_index=True)
            else:
                # if sequence, prepare sequence dataframe
                if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
                    # prepare dataframe for sequence model
                    video_path_crops_real = os.path.join(
                        dataset_path + "/facecrops/real/")
                    video_path_crops_fake = os.path.join(
                        dataset_path + "/facecrops/fake/")

                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video})

                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    df = prepare_sequence_data(dataset, df)
                    # add path to data
                    for idx, row in df.iterrows():
                        if row['label'] == 0:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_real) + str(row['original'])
                        elif row['label'] == 1:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_fake) + str(row['original'])
                else:
                    # if face crops available go to path with face crops
                    video_path_crops_real = os.path.join(
                        dataset_path + "/facecrops/real/")
                    video_path_crops_fake = os.path.join(
                        dataset_path + "/facecrops/fake/")
                    # add labels to videos
                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video_path_crops_real + video})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video_path_crops_fake + video})
                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    if len(df) == 0:
                        raise ValueError(
                            "No faces available. Please set faces_available=False.")
        

    else:
        # prepare test data
        if dataset == 'celebdf':
            # reading in the celebdf testing data
            df_test = pd.read_csv(
                dataset_path + '/List_of_testing_videos.txt', sep=" ", header=None)
            df_test.columns = ["label", "video"]
            # switch labels so that fake label is 1
            df_test['label'] = df_test['label'].apply(switch_one_zero)
            df_test['video'] = dataset_path + '/' + df_test['video']
            print(f"{len(df_test)} test videos.")
            return df_test
        # put data into dataframe
        df = pd.DataFrame(data=data_list)

    if test_data:
        print(f"{len(df)} test videos.")
    else:
        if face_crops:
            print(f"Lead to: {len(df)} face crops.")
        else:
            print(f"{len(df)} train videos.")
    print()
    return df


def df_augmentations(img_size, strength="weak"):
    """
    Augmentations with the albumentations package.
    # Arguments:
        strength: strong or weak augmentations

    # Implementation: Christopher Otto
    """
    if strength == "weak":
        print("Weak augmentations.")
        augs = Compose([
            # hflip with prob 0.5
            HorizontalFlip(p=0.5),
            # adjust image to DNN input size
            Resize(width=img_size, height=img_size)
        ])
        return augs
    elif strength == "strong":
        print("Strong augmentations.")
        # augmentations via albumentations package
        # augmentations adapted from Selim Seferbekov's 3rd place private leaderboard solution from
        # https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721
        augs = Compose([
            # hflip with prob 0.5
            HorizontalFlip(p=0.5),
            ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            GaussNoise(p=0.1),
            GaussianBlur(blur_limit=3, p=0.05),
            PadIfNeeded(min_height=img_size, min_width=img_size,
                        border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(),
                   HueSaturationValue()], p=0.7),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                             rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            # adjust image to DNN input size
            Resize(width=img_size, height=img_size)
        ])
        return augs
    else:
        raise ValueError(
            "This augmentation option does not exist. Choose \"weak\" or \"strong\".")


def structure_uadfv_files(path_to_data):
    """Creates test folders and moves test videos there."""
    os.mkdir(path_to_data + '/test/')
    os.mkdir(path_to_data + '/test/fake/')
    os.mkdir(path_to_data + '/test/real/')
    test_data = pd.read_csv(
        os.getcwd() + "/deepfake_detector/data/uadfv_test.csv", names=['video'], header=None)
    for idx, row in test_data.iterrows():
        if len(str(row.loc['video'])) > 8:
            # video is fake, therefore copy it into fake test folder
            shutil.copy(path_to_data + '/fake/' +
                        row['video'], path_to_data + '/test/fake/')
        else:
            # video is real, therefore move it into real test folder
            shutil.copy(path_to_data + '/real/' +
                        row['video'], path_to_data + '/test/real/')


def reproducibility_seed(seed):
    print(f"The random seed is set to {seed}.")
    # set numpy random seed
    np.random.seed(seed)
    # set pytorch random seed for cpu and gpu
    torch.manual_seed(seed)
    # get deterministic behavior
    torch.backends.cudnn.deterministic = True


def switch_one_zero(num):
    """Switch label 1 to 0 and 0 to 1
        so that fake videos have label 1.
    """
    if num == 1:
        num = 0
    else:
        num = 1
    return num


def prepare_sequence_data(dataset, df):
    """
    Prepares the dataframe for sequence models.
    """
    print(df)
    df = df.sort_values(by=['video']).reset_index(drop=True)
    # add original column
    df['original'] = ""
    if dataset == 'celebdf':
        print("Preparing sequence data.")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # remove everything after last underscore
            df.loc[idx, 'original'] = row.loc['video'].rpartition("_")[0]
    # count frames per video
    df1 = df.groupby(['original']).size().reset_index(name='count')
    df = pd.merge(df, df1, on='original')
    # remove videos that don't where less than 20 frames
    # were detected to ensure equal frame size of 20 for sequence
    # for dfdc only 5 frames because dataset is so large
    if dataset == 'dfdc':
        df = df[df['count'] == 5]
    else:
        df = df[df['count'] == 20]
    df = df[['label', 'original']]
    # ensure that dataframe includes each video with 20 frames once
    df = df.groupby(['label', 'original']).size().reset_index(name='count')
    df = df[['label', 'original']]
    return df


def setup_celebdf_benchmark(data_path, method):
    """
    Setup the folder structure of the Celeb-DF Dataset.
    """
    if data_path is None:
        raise ValueError("""Please go to https://github.com/danmohaha/celeb-deepfakeforensics
                                and scroll down to the dataset section.
                                Click on the link \"this form\" and download the dataset. 
                                Extract the files and organize the folders follwing this folder structure:
                                ./celebdf/
                                        Celeb-real/
                                        Celeb-synthesis/
                                        YouTube-real/
                                        List_of_testing_videos.txt
                                """)
    if data_path.endswith("celebdf"):
        print(
            f"Benchmarking \033[1m{method}\033[0m on the \033[1m Celeb-DF \033[0m dataset with ...")
    else:
        raise ValueError("""Please organize the dataset directory in this way:
                            ./celebdf/
                                    Celeb-real/
                                    Celeb-synthesis/
                                    YouTube-real/
                                    List_of_testing_videos.txt
                        """)


def process_directory(directory_path, label, method, cmd):
    true_labels = []
    predicted_probs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".mp4"):  # or other video formats
            video_path = os.path.join(directory_path, filename)
            predicted_prob = DFDetector.detect_single(
                video_path=video_path, method=method, cmd=cmd)
            if predicted_prob != -1:  # Skip videos where no faces were detected
                true_labels.append(label)
                predicted_probs.append(predicted_prob)
            else:
                print(f"Skipping video {video_path} due to no face detection.")
    return true_labels, predicted_probs

def calculate_cumulative_auc(file_path):
    true_labels = []
    predicted_probs = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                try:
                    data = json.loads(line)
                    true_labels.extend(data['true_labels'])
                    predicted_probs.extend(data['predicted_probs'])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")

    if true_labels and predicted_probs:
        return roc_auc_score(true_labels, predicted_probs)
    return 0


def main():

    results_file = "results.txt"
    # parse arguments
    args = parser.parse_args()
    # initialize the deepfake detector with the desired task
    if args.detect_single:
        true_labels = []
        predicted_probs = []
        if args.real_videos_path and args.fake_videos_path:
            # Process real videos
            real_labels, real_probs = process_directory(args.real_videos_path, label=0, method=args.detection_method, cmd=args.cmd)
            true_labels.extend(real_labels)
            predicted_probs.extend(real_probs)

            # Process fake videos
            fake_labels, fake_probs = process_directory(args.fake_videos_path, label=1, method=args.detection_method, cmd=args.cmd)
            true_labels.extend(fake_labels)
            predicted_probs.extend(fake_probs)

            # Store results in a text file
            with open(results_file, 'a') as file:
                result = {
                    'true_labels': true_labels,
                    'predicted_probs': predicted_probs
                }
                file.write(json.dumps(result) + '\n')

        else:
            print("Both real_videos_path and fake_videos_path must be provided.")
            return

        # Calculate and print AUC score
        if true_labels and predicted_probs:
            auc_score = roc_auc_score(true_labels, predicted_probs)
            print(f"AUC score: {auc_score}")

            # Calculate cumulative AUC score
            cumulative_auc_score = calculate_cumulative_auc(results_file)
            print(f"Cumulative AUC score: {cumulative_auc_score}")
    elif args.benchmark:
        DFDetector.benchmark(
            dataset=args.dataset, data_path=args.data_path, method=args.detection_method)
    elif args.train:
        print(args)
        print(args.facecrops_available)
        DFDetector.train_method(dataset=args.dataset, data_path=args.data_path, method=args.model_type, img_save_path=args.save_path, epochs=args.epochs, batch_size=args.batch_size,
                     lr=args.lr, folds=args.folds, augmentation_strength=args.augs, fulltrain=args.fulltrain,  face_margin=args.face_margin, faces_available=args.facecrops_available, seed=args.seed)
    else:
        print("Please choose one of the three modes: detect_single, benchmark, or train.")

if __name__ == "__main__":
    main()
