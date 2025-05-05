import sys
sys.setrecursionlimit(15000)
import os
import torch
import numpy as np
from torch.autograd import Variable
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from PIL import Image
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score
import math
import model_big
import cv2  # For video processing

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='databases/faceforensicspp/test', help='path to test dataset')
parser.add_argument('--real', default='0_original', help='real folder name')
parser.add_argument('--deepfakes', default='1_deepfakes', help='fake folder name')
parser.add_argument('--batchSize', type=int, default=10, help='batch size')
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--outf', default='checkpoints/binary_faceforensicspp', help='folder to output images and model checkpoints')
parser.add_argument('--random_sample', type=int, default=0, help='number of random sample to test')
parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')
parser.add_argument('--id', type=int, default=21, help='checkpoint ID')
parser.add_argument('--results_file', default='results.txt', help='file to store results for AUC calculation')

opt = parser.parse_args()
print(opt)

def get_video_list(path):
    video_lst = []

    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            if f.lower().endswith(('mp4', 'avi', 'mov', 'mkv')):
                video_lst.append(f)

    return video_lst

def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(transform_fwd(frame).unsqueeze(0))
    cap.release()
    return frames

def classify_batch(vgg_ext, model, batch):
    n_sub_imgs = len(batch)

    if (opt.random_sample > 0):
        if n_sub_imgs > opt.random_sample:
            np.random.shuffle(batch)
            n_sub_imgs = opt.random_sample

        img_tmp = torch.FloatTensor([]).view(0, 3, opt.imageSize, opt.imageSize)

        for i in range(n_sub_imgs):
            img_tmp = torch.cat((img_tmp, batch[i]), dim=0)

        if opt.gpu_id >= 0:
            img_tmp = img_tmp.cuda(opt.gpu_id)

        input_v = Variable(img_tmp, requires_grad = False)

        x = vgg_ext(input_v)
        classes, class_ = model(x, random=opt.random)
        output_pred = class_.data.cpu().numpy()

    else:
        batchSize = opt.batchSize
        steps = int(math.ceil(n_sub_imgs*1.0/batchSize))

        output_pred = np.array([], dtype=float).reshape(0,2)

        for i in range(steps):
            img_tmp = torch.FloatTensor([]).view(0, 3, opt.imageSize, opt.imageSize)

            end = (i + 1)*batchSize
            if end > n_sub_imgs:
                end = n_sub_imgs - i*batchSize
            else:
                end = batchSize

            for j in range(end):
                img_tmp = torch.cat((img_tmp, batch[i*batchSize + j]), dim=0)

            if opt.gpu_id >= 0:
                img_tmp = img_tmp.cuda(opt.gpu_id)

            input_v = Variable(img_tmp, requires_grad = False)

            x = vgg_ext(input_v)
            classes, class_ = model(x, random=opt.random)
            output_p = class_.data.cpu().numpy()

            output_pred = np.concatenate((output_pred, output_p), axis=0)

    output_pred = output_pred.mean(0)

    if output_pred[1] >= output_pred[0]:
        pred = 1
    else:
        pred = 0

    return pred, output_pred[1]

transform_fwd = transforms.Compose([
    transforms.Resize((opt.imageSize, opt.imageSize)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def classify_video(vgg_ext, model, video_path, label):
    frames = extract_frames_from_video(video_path)
    if len(frames) == 0:
        return 0, 0, np.array([]), np.array([])

    cls, prob = classify_batch(vgg_ext, model, frames)
    correct = 1 if cls == label else 0

    return 1, correct, np.array([label]), np.array([prob])

def save_results(results_file, labels, probs):
    with open(results_file, 'a') as f:
        for label, prob in zip(labels, probs):
            f.write(f"{label} {prob}\n")

def calculate_auc_from_file(results_file):
    labels = []
    probs = []
    with open(results_file, 'r') as f:
        for line in f:
            label, prob = map(float, line.strip().split())
            labels.append(label)
            probs.append(prob)
    
    labels = np.array(labels)
    probs = np.array(probs)
    
    fpr, tpr, _ = roc_curve(labels, probs, pos_label=1)
    auc_score = roc_auc_score(labels, probs)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    print('##################################')
    print('EER: %.2f' % (eer * 100))
    print('AUC: %.4f' % auc_score)

    return auc_score

if __name__ == '__main__':
    path_real = os.path.join(opt.dataset, opt.real)
    path_deepfakes = os.path.join(opt.dataset, opt.deepfakes)

    vgg_ext = model_big.VggExtractor()
    model = model_big.CapsuleNet(2, opt.gpu_id)

    model.load_state_dict(torch.load(os.path.join(opt.outf, 'capsule_' + str(opt.id) + '.pt')))
    model.eval()

    if opt.gpu_id >= 0:
        vgg_ext.cuda(opt.gpu_id)
        model.cuda(opt.gpu_id)

    for epoch in range(1):  # Replace with the number of epochs you have
        print(f"Epoch {epoch + 1}")
        ###################################################################
        tol_count_vid_real = 0
        tol_correct_real = 0

        tol_count_vid_deepfakes = 0
        tol_correct_deepfakes = 0

        # real data
        real_videos = get_video_list(path_real)
        label_lst_global = np.array([], dtype=float)
        prob_lst_global = np.array([], dtype=float)

        print("Processing real videos...")
        for video in tqdm(real_videos, desc="Real Videos"):
            count_vid, correct, label_lst, prob_lst = classify_video(vgg_ext, model, os.path.join(path_real, video), 0)
            tol_count_vid_real += count_vid
            tol_correct_real += correct
            label_lst_global = np.concatenate((label_lst_global, label_lst), axis=0)
            prob_lst_global = np.concatenate((prob_lst_global, prob_lst), axis=0)

        # deepfakes data
        deepfake_videos = get_video_list(path_deepfakes)

        print("Processing deepfake videos...")
        for video in tqdm(deepfake_videos, desc="Deepfake Videos"):
            count_vid, correct, lb_lst, pb_lst = classify_video(vgg_ext, model, os.path.join(path_deepfakes, video), 1)
            tol_count_vid_deepfakes += count_vid
            tol_correct_deepfakes += correct

            label_lst_global = np.concatenate((label_lst_global, lb_lst), axis=0)
            prob_lst_global = np.concatenate((prob_lst_global, pb_lst), axis=0)

        tol_count_vid = tol_count_vid_real + tol_count_vid_deepfakes
        tol_correct = tol_correct_real + tol_correct_deepfakes

        print('##################################')
        print('Number of videos: %d' % (tol_count_vid))
        print('Number of correct classifications: %d' % (tol_correct))
        print('Accuracy: %.2f' % (tol_correct / tol_count_vid * 100))

        print('##################################')
        print('Number of correct real classifications: %d' % (tol_correct_real))
        print('Accuracy: %.2f' % (tol_correct_real / tol_count_vid_real * 100))

        print('##################################')
        print('Number of correct deepfakes classifications: %d' % (tol_correct_deepfakes))
        print('Accuracy: %.2f' % (tol_correct_deepfakes / tol_count_vid_deepfakes * 100))

        label_lst_global[label_lst_global > 0] = 1

        # Save the results to the file
        save_results(opt.results_file, label_lst_global, prob_lst_global)

        # Calculate and print the AUC score from the file
        auc = calculate_auc_from_file(opt.results_file)
        print(f"Cumulative AUC after epoch {epoch + 1}: {auc:.4f}")
