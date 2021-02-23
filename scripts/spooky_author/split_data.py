import argparse

import pandas
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description='Split the spooky author dataset into two files, train and test.')
    parser.add_argument(
        '--full-path', type=str, help='Path of full dataset', required=True)
    parser.add_argument(
        '--train-path', type=str, help='Path to write training dataset', required=True)
    parser.add_argument(
        '--test-path', type=str, help='Path to write test dataset', required=True)
    args = parser.parse_args()

    data = pandas.read_csv(args.full_path, index_col='id')
    train_data, test_data = train_test_split(
        data, train_size=0.75, shuffle=True, random_state=1)

    # Store splits
    train_data.to_csv(args.train_path, index=True)
    print('Training data written to:', args.train_path)

    test_data.to_csv(args.test_path, index=True)
    print('Test data written to:', args.test_path)


if __name__ == '__main__':
    main()
