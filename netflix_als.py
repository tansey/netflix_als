'''
Implements Alternating Least Squares (ALS) to create a recommender system for a subset of the Netflix dataset.
'''
import numpy as np
import argparse
import scipy.io

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs Alternating Least Squares (ALS) to learn latent features for users and movies.')

    parser.add_argument('infile', help='The netflix dataset subset to use.')

    parser.add_argument('--verbose', type=int, default=0, help='Print detailed progress information to the console. 0=none, 1=high-level info, 2=all details.')
    parser.add_argument('--solver', choices=['cf', 'cd'], default='cf', help='The method used to solve alternating least squares problem. cf=closed form. cd=coordinate descent.')
    parser.add_argument('--outfile', default='results.csv', help='The file where the resulting matrix will be saved.')

    # Get the arguments from the command line
    args = parser.parse_args()

    # Load data from file
    data = scipy.io.loadmat(args.infile)

    print 'matrix: {0}'.format(data)