import numpy as np


class HexBoard:
    def __init__(self, w: int, h: int, initial: np.ndarray = None):
        self.w = w
        self.h = h

        if initial is None:
            initial = np.zeros((h, w, 1), int)

        self.board = initial

    def reset_board(self):
        self.board = np.zeros((self.h, self.w, 1), int)

    def __str__(self):
        string = ""

        for i, row in enumerate(self.board):
            string += "  " * i
            string += "  ".join(f"{n:2d}" for n in row) + "\n"

        return string

    #def __str__(self):
    #    string = ""
    #
    #    for i in range(self.w - 1, -self.h, -1):
    #        string += "  " * abs(i)
    #        string += "  ".join(map(lambda n: f"{n:2d}", self.board.diagonal(i, 0).flatten()))
    #        string += "\n"
    #
    #    return string


def from_np_array(w: int, h: int, data: np.ndarray):
    return HexBoard(w, h, initial=data.reshape(h, w))
