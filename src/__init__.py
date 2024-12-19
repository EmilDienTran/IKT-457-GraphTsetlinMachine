import math
from time import time
from collections import deque
from numpy import number

from src.data import dataset as dl
from src.game import hex_board
import numpy as np
from src.tsetlinmachine.graphs import Graphs
from src.tsetlinmachine.tm import GraphTsetlinMachine, MultiClassGraphTsetlinMachine
import argparse
from datetime import datetime

from src.helperfunctions import draw_simple_graph, show_graph_nodes, show_graph_edges

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=200, type=int)
    parser.add_argument("--T", default=400, type=int)
    parser.add_argument("--s", default=1.2, type=float)
    parser.add_argument("--depth", default=6, type=int)
    parser.add_argument("--hypervector-size", default=2048, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=2048, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=64, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args



args = default_args()

def get_cell_info(board, i, j, board_dim):
    node_name = f"cell_{i}_{j}"
    index = i * board_dim + j

    if board[index] == 1:
        color = "white"
    elif board[index] == -1:
        color = "black"
    else:
        color = "empty"

    if i == 0 and j == 0:
        location = "top_left"
    elif i == 0 and j == board_dim - 1:
        location = "top_right"
    elif i == board_dim - 1 and j == 0:
        location = "bottom_left"
    elif i == board_dim - 1 and j == board_dim - 1:
        location = "bottom_right"
    elif i == 0:
        location = "top_edge"
    elif i == board_dim - 1:
        location = "bottom_edge"
    elif j == 0:
        location = "left_edge"
    elif j == board_dim - 1:
        location = "right_edge"
    else:
        location = "interior"

    points = "placeholder"

    dist_to_top = i
    dist_to_bottom = board_dim - 1 - i
    dist_to_left = j
    dist_to_right = board_dim - 1 - j

    found_chain, chain_pos, connections = find_chain(i, j, board_dim, board, color)


    chain_len = 0
    if len(chain_pos) < 7:
        chain_len = len(chain_pos)
    else:
        chain_len = 7

    node_properties = {
        "node_name": node_name,
        "color": color,
        "location": location,
        "points": points,
        "distance_top": get_distance_category(dist_to_top, board_dim),
        "distance_bottom": get_distance_category(dist_to_bottom, board_dim),
        "distance_left": get_distance_category(dist_to_left, board_dim),
        "distance_right": get_distance_category(dist_to_right, board_dim),
        "distance_center": get_distance_category(dist_to_top, board_dim),
        "distance_diagonal": get_distance_from_diagonal(i,j, board_dim),
        "chain_type": get_chain_info(color, found_chain, connections),
        "number_of_neighbors": get_number_of_neighbors(board, i, j, board_dim),
        "chain_pos": f'chain_{chain_len}',
    }

    return node_properties

def get_edge_info(board, i, j, board_dim, node_properties):
    number_of_edges = 0
    neighbor_cell = []

    direction_names = {
        (-1, 0): "northwest",  # Up
        (-1, 1): "northeast",  # Up-Right
        (0, -1): "west",  # Left
        (0, 1): "east",  # Right
        (1, -1): "southwest",  # Down-Left
        (1, 0): "southeast"  # Down
    }
    for vector_x, vector_y in direction_names.keys():
        new_pos_x, new_pos_y = i + vector_x, j + vector_y

        neighbor_information = {
            "direction": direction_names[(vector_x, vector_y)],
            "coordinates": (new_pos_x, new_pos_y)
        }
        if 0 <= new_pos_x < board_dim and 0 <= new_pos_y < board_dim:
            number_of_edges += 1

            neighbor_properties = get_cell_info(board, new_pos_x, new_pos_y, board_dim)
            edge_type = get_node_connection_type(neighbor_properties, node_properties, i, j, board_dim, board)


            neighbor_information.update({
                "inside_board": True,
                "properties": neighbor_properties,
                "edge_type": edge_type
            })
        else:
            neighbor_information.update({
                "inside_board": False,
                "properties": {
                    "color": "goofy",
                    "location": "outside_board"
                },
                "edge_type": "who cares"
            })

        neighbor_cell.append(neighbor_information)

    edge_properties = {
        "number_of_edges": number_of_edges,
        "current_node": node_properties,
        "neighbors": neighbor_cell,
    }

    return edge_properties

# TODO: We should maybe change this to handle more explicit edges, such as blocking the opponent.
def get_node_connection_type(neighbor_properties, node_properties, i, j, board_dim, board):
    neighbor_color = neighbor_properties["color"]
    node_color = node_properties["color"]

    if neighbor_color == "empty" and node_color == "empty":
        return "empty_connection"

    if neighbor_color == "empty" or node_color == "empty":
        return "expansion"

    if neighbor_color == "black" and node_color == "white":
        return "black_white_connection"

    if neighbor_color == 'white' and node_color == "black":
        return "white_black_connection"

    if neighbor_color == node_color:
        if node_properties["chain_type"] == "winning" or neighbor_properties["chain_type"] == "winning":
            return "winning_chain"
        return "connection"

    return "blocking"

    #TODO: Edge type bridge
    #TODO: Winning chain with bridge
    #TODO: next turn counter?
    #TODO: Absolute chain length in winning axis

# TODO: Find strategic patterns
def find_strategic_patterns(board, i, j, board_dim, node_properties, edge_properties):
    pass

# TODO: Find strategic positions - Middle is better than edges
def find_strategic_position_value(board, i, j, board_dim):
    pass
    #Each node gets a certain amount of points for strategic placement
    # for example middle position (4, 3) gets 3 points, edges get 2 or 1
    # Corners get 4-5 points
    #should it be diffrent for each player?

# TODO: Chain links - Chain length in winning axis

def find_chain(i, j, board_dim, board, chain_color):

    direction_vectors = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    if board[i * board_dim + j] == 1 and chain_color == "white": #White
        connects_start = i == 0
        connects_end = i == board_dim - 1
    elif board[i * board_dim + j] == -1 and chain_color == "black": #Black
        connects_start = j == 0
        connects_end = j == board_dim - 1
    else:  # empty
        return False, set(), {"connects_start": False, "connects_end": False}

    visited = {(i, j)}
    to_visit = deque([(i, j)])


    while to_visit:
        current_pos_i, current_pos_j = to_visit.popleft()

        for vector_x, vector_y in direction_vectors:
            new_pos_x, new_pos_y = current_pos_i + vector_x, current_pos_j + vector_y
            if 0 <= new_pos_x < board_dim and 0 <= new_pos_y < board_dim and (new_pos_x, new_pos_y) not in visited:

                if board[new_pos_x * board_dim + new_pos_y] == 1 and chain_color == "white":
                    connects_start = connects_start or new_pos_x == 0
                    connects_end = connects_end or new_pos_x == board_dim - 1
                    visited.add((new_pos_x, new_pos_y))
                    to_visit.append((new_pos_x, new_pos_y))
                elif board[new_pos_x * board_dim + new_pos_y] == -1 and chain_color == "black":
                    connects_start = connects_start or new_pos_y == 0
                    connects_end = connects_end or new_pos_y == board_dim - 1
                    visited.add((new_pos_x, new_pos_y))
                    to_visit.append((new_pos_x, new_pos_y))

    found_chain = len(visited) > 1

    connections = {"connects_start": connects_start, "connects_end": connects_end}


    return found_chain, visited, connections

def get_chain_info(color, found_chain, connections):

    chain_type = "no_chain"
    if color == "empty":
        chain_type = "no_chain"
    elif found_chain:
        if connections["connects_start"] and connections["connects_end"]:
            chain_type = "winning"
        else:
            chain_type = "small_chain"

    return chain_type

def get_distance_category(dist, board_dim):
    if dist == 0:
        return "edge"
    elif dist <= board_dim // 3:
        return "near"
    elif dist <= 2 * board_dim // 3:
        return "mid"
    else:
        return "far"

def get_distance_from_center(board, i, j, board_dim):
    center = board_dim // 2
    current_dist_from_center = abs(i - center) + abs(j - center)
    if current_dist_from_center <= 1:
        return "center"
    elif current_dist_from_center <= 2:
        return "near_center"
    else:
        return "outer"

def get_distance_from_diagonal(i, j, board_dim):
    main_diag_dist = abs(i - j)
    anti_diag_dist = abs(i + j - (board_dim - 1))
    min_dist = min(main_diag_dist, anti_diag_dist)
    if min_dist == 0:
        return "on_diagonals"
    elif min_dist <= 1:
        return "adjacent_diagonals"
    else:
        return "far_diagonals"

def get_number_of_neighbors(board, i, j, board_dim):
    direction_vectors = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
    white_neighbor_count = 0
    black_neighbor_count = 0
    empty_neighbor_count = 0
    for vector_x, vector_y in direction_vectors:
        new_pos_x, new_pos_y = i + vector_x, j + vector_y
        if board[new_pos_x * board_dim + new_pos_y] == 1 and board[i * board_dim + j] == 1:
            white_neighbor_count += 1
        elif board[new_pos_x * board_dim + new_pos_y] == -1 and board[i * board_dim + j] == -1:
            black_neighbor_count += 1
        else:
            empty_neighbor_count += 1

        if empty_neighbor_count > black_neighbor_count and empty_neighbor_count > white_neighbor_count:
            return "empty_neighbor_majority"
        else:
            if white_neighbor_count > black_neighbor_count:
                return "white_neighbour_majority"
            elif black_neighbor_count > white_neighbor_count:
                return "black_neighbour_majority"
            elif black_neighbor_count == white_neighbor_count:
                return "same_number_of_neighbours"

def get_symbols(board_dim):
    symbols = []

    symbols.extend(["white", "black", "empty"])

    symbols.extend([
        "top_left", "top_right", "bottom_left", "bottom_right",
        "top_edge", "bottom_edge", "left_edge", "right_edge",
        "interior", "edge", "near", "mid", "far"
    ])

    symbols.extend(["center", "near_center", "outer"])
    symbols.extend(["on_diagonals", "adjacent_diagonals", "far_diagonals"])
    symbols.extend(["empty_neighbor_majority", "white_neighbour_majority","black_neighbour_majority","same_number_of_neighbours"])
    symbols.extend(["no_chain", "winning", "small_chain"])

    return symbols

def main():

    current_time = datetime.now().replace(microsecond=0).strftime("%Y-%m-%d %H-%M-%S")
    log_file = f"./src/logs/epochs-{current_time}.log"
    dataset_file = "./src/data/datasets/hex_games.csv"

    dataset = dl.Dataset(dataset_file, train_fraction=0.9)
    (x_train, y_train), (x_test, y_test) = dataset.get_train_data(), dataset.get_test_data()

    boardpres = hex_board.from_np_array(int(math.sqrt(x_train.shape[1])), int(math.sqrt(x_train.shape[1])), x_train[0])
    print("First Board(Graph_0):")
    print(boardpres)

    print("\nCreating training data")
    graph_train = convertDataToGraphs(x_train)

    print("\nCreating testing data")
    graph_test = convertDataToGraphs(x_test, init_graphs=graph_train)

    print("\nRunning helper function 1 (draw_simple_graph)")
    print(f"\nSaved file plotgraph-{current_time}.png in src/outputs")
    draw_simple_graph(graph_train, 0, f"plotgraph-{current_time}.png")

    print("\nRunning helper function 2 (show_graph_nodes)")
    show_graph_nodes(graph_train, 0)

    print("\nRunning helper function 3 (show_graph_edges)")
    show_graph_edges(graph_train, 0)


    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        depth=args.depth,
        message_size=args.message_size,
        message_bits=args.message_bits,
        max_included_literals=args.max_included_literals,
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1)
    )

    with open(log_file, "w") as file:
        file.write(f"Log start time: {current_time}\n\n"
                   f"Dataset board dimensions: {int(math.sqrt(x_train.shape[1]))} X {int(math.sqrt(x_train.shape[1]))}\n"
                   f"Dataset size: {x_train.shape[0] + x_test.shape[0]}\n"
                   f"Train | Test split: {x_train.shape[0]} | {x_test.shape[0]}\n"
                   f"===========================================================\n"
                   f"Tsetlin Machine Hyperparameters: \n"
                   f"epochs = {args.epochs}\n"
                   f"number of clauses = {args.number_of_clauses}\n"
                   f"T = {args.T}\n"
                   f"s = {args.s}\n"
                   f"===========================================================\n"
                   f"Accuracy per epoch: \n")

    print("\nTraining started")
    for i in range(args.epochs):
        start_training = time()
        tm.fit(graph_train, y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result_test = 100 * (tm.predict(graph_test) == y_test).mean()
        stop_testing = time()

        result_train = 100 * (tm.predict(graph_train) == y_train).mean()

        print("%d %.2f %.2f %.2f %.2f" % (
            i, result_train, result_test, stop_training - start_training, stop_testing - start_testing))

        with open(log_file, "a") as file:
            file.write("%d %.2f %.2f %.2f %.2f\n" % (
                i, result_train, result_test, stop_training - start_training, stop_testing - start_testing
            ))


def convertDataToGraphs(X, init_graphs=None):
    board_dim = int(math.sqrt(X.shape[1]))

    graphs = Graphs(
        X.shape[0],
        symbols=get_symbols(board_dim),
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
        double_hashing=args.double_hashing,
        init_with=init_graphs
    )

    for graph_id, board in enumerate(X):
        graphs.set_number_of_graph_nodes(graph_id, board_dim * board_dim)

    graphs.prepare_node_configuration()
    for graph_id, board in enumerate(X):
        for i in range(board_dim): # y
            for j in range(board_dim): # x

                node_name = f"cell_{i}_{j}"
                current_node_properties = get_cell_info(board, i, j, board_dim)
                edge_info = get_edge_info(board, i, j, board_dim, current_node_properties)

                graphs.add_graph_node(graph_id, node_name, edge_info["number_of_edges"])

                graphs.add_graph_node_property(graph_id, node_name, current_node_properties["color"])
                graphs.add_graph_node_property(graph_id, node_name, current_node_properties["location"])
                graphs.add_graph_node_property(graph_id, node_name, current_node_properties["chain_type"])

                graphs.add_graph_node_property(graph_id, node_name, current_node_properties["distance_top"])
                graphs.add_graph_node_property(graph_id, node_name, current_node_properties["distance_bottom"])
                graphs.add_graph_node_property(graph_id, node_name, current_node_properties["distance_left"])
                graphs.add_graph_node_property(graph_id, node_name, current_node_properties["distance_right"])
                graphs.add_graph_node_property(graph_id, node_name, current_node_properties["distance_center"])
                graphs.add_graph_node_property(graph_id, node_name, current_node_properties["distance_diagonal"])
                graphs.add_graph_node_property(graph_id, node_name, current_node_properties["number_of_neighbors"])


    graphs.prepare_edge_configuration()

    for graph_id, board in enumerate(X):
        for i in range(board_dim):
            for j in range(board_dim):
                current_node_properties = get_cell_info(board, i, j, board_dim)
                edge_info = get_edge_info(board, i, j, board_dim, current_node_properties)

                for neighbor in edge_info["neighbors"]:
                    if neighbor["inside_board"]:
                        new_pos_x, new_pos_y = neighbor["coordinates"]
                        graphs.add_graph_node_edge(graph_id, current_node_properties["node_name"],f"cell_{new_pos_x}_{new_pos_y}",neighbor["edge_type"])

    graphs.encode()
    print(f"Number of nodes per graph: {graphs.number_of_graph_nodes[0]}")
    print(f"Total number of edges: {graphs.edge.shape[0]}\n")
    return graphs


if __name__ == "__main__":
    main()