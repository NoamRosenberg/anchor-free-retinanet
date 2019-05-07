import argparse
import pickle
import os

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--eval_path', default='/data/deeplearning/dataset/training/data/new_loss_evaluations')
    parser.add_argument('--model', default='')
    parser = parser.parse_args(args)

    for eval_file in os.listdir(parser.eval_path):
        if parser.model in eval_file:
            with open(os.path.join(parser.eval_path,eval_file),'rb') as f:
                data = pickle.load(f)
                print('for {}'.format(eval_file))
                data.summarize()
                print('***********************************************')


if __name__ == '__main__':
    main()