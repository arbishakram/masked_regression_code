import argparse
import os
from solver import mr_solver

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_dataset_dir', dest='train_dataset_dir', default='dataset/', help='path of the train dataset')
parser.add_argument('--test_dataset_dir', dest='test_dataset_dir', default='dataset/', help='path of the test dataset')
parser.add_argument('--inthewild_dataset_dir', dest='inthewild_dataset_dir', default='', help='path of the inthewild test dataset')
parser.add_argument('--image_size', dest='image_size', type=int, default=128, help='size of image')
parser.add_argument('--total_images', dest='total_images', type=int, default=200, help='total images in trainA')
parser.add_argument('--input_ch', dest='input_ch', type=int, default=3, help='# of input image channels')
parser.add_argument('--receptive_field', dest='receptive_field', type=int, default=3, help='size of receptive field')
parser.add_argument('--lamda', dest='lamda', type=float, default=0.4, help='lambda value')
parser.add_argument('--mode', dest='mode', default='train', help='train, test, test_inthewild')
parser.add_argument('--weights_dir', dest='weights_dir', default='Exp_N2H/weights/', help='weights are saved here')
parser.add_argument('--test_results_dir', dest='test_results_dir', default='Exp_N2H/test_results/', help='test results are saved here')
parser.add_argument('--inthewild_results_dir', dest='inthewild_results_dir', default='Exp_N2H/inthewild_results/', help='inthewild results are saved here')
args = parser.parse_args()

def main():
   
    # Create directories if not exist.
    if not os.path.exists(args.weights_dir):
        os.makedirs(args.weights_dir)
    if not os.path.exists(args.test_results_dir):
        os.makedirs(args.test_results_dir)
    if not os.path.exists(args.inthewild_results_dir):
        os.makedirs(args.inthewild_results_dir)
  
    # Solver for training and testing MR.
    model = mr_solver(args)
    model.train(args) if args.mode == 'train' \
        else model.test(args) 

main()
