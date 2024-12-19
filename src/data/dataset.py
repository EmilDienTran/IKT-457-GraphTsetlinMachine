import numpy as np
from zipfile import ZipFile


class Dataset:
    def __init__(self, file_path, train_fraction=0.5):

        print(f"Loading dataset {file_path}", end="\n")
        self.dataset = self.load_dataset(file_path)
        print(f"Loaded dataset {file_path}", end="\n")

        self.boards = self.dataset[:, :-1]
        self.labels = self.dataset[:, -1:].flatten()
        self.labels = np.where(self.labels == 1, 1, 0)

        split_index = int(len(self.boards) * train_fraction)
        self.train_boards = self.boards[:split_index]
        self.train_labels = self.labels[:split_index]
        self.test_boards = self.boards[split_index:]
        self.test_labels = self.labels[split_index:]

    def load_dataset(self, file_path):
        if file_path.endswith((".txt", ".csv")):
            return np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=int)

        elif file_path.endswith(".zip"):
            with ZipFile(file_path) as archive:
                for filename in archive.namelist():
                    if filename.endswith((".txt", ".csv")):
                        return np.genfromtxt(archive.open(filename), delimiter=',', skip_header=1, dtype=int)
        return None

    def __len__(self):
        return self.boards.shape[0]

    def get_train_data(self):
        return self.train_boards, self.train_labels

    def get_test_data(self):
        return self.test_boards, self.test_labels

    def get_data(self):
        return self.get_train_data(), self.get_test_data()
