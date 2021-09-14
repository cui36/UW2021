# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:13:33 2020

@author: Yun-Hsuan Su (June 19, 2018), modified by Yana Sosnovskaya (July 2020)
"""

# ref: https://www.laurentluce.com/posts/solving-mazes-using-python-simple-recursivity-and-a-search/
# ref: http://code.activestate.com/recipes/577519-a-star-shortest-path-algorithm/

from __future__ import print_function

import matplotlib.pyplot as plt


class AStarGraph(object):
    # Define a class board like grid with two barriers

    def __init__(self):
        self.barriers = []
    # uncomment a line appropriate to your team number
    # for groups 1, 2, 3, 4 uncomment next line:
        self.barriers.append([(2, 4), (2, 5), (2, 6), (3, 6), (4, 6), (5, 6), (5, 5), (5, 4), (5, 3), (5, 2), (4, 2), (3, 2)])
    # for groups 5, 6, 7, 8, 9, 10 uncomment next line:
    # self.barriers.append([(2, 3), (1, 5), (2, 7), (3, 6), (4, 6), (5, 6), (5, 5), (5, 4), (5, 3), (5, 2), (4, 2), (3, 2)])

    def heuristic(self, start, goal):
        # Use Chebyshev distance heuristic if we can move one square either
        # adjacent or diagonal
        # https://lyfat.wordpress.com/2012/05/22/euclidean-vs-chebyshev-vs-manhattan-distance/

        current_heuristic = -1
        # TODO: -------------------------------------------------
        # you may be using: start[0], start[1], goal[0], goal[1], abs(), max(,), min(,)

        dx = abs(start[0] - goal[0])
        dy = abs(start[1] - goal[1])
        current_heuristic = max(dx, dy)
        # -------------------------------------------------------
        #print(current_heuristic)
        return current_heuristic

    def get_vertex_neighbours(self, pos):
        n = []
        # print(pos[0])
        # print(pos[1])
        # Moves allow link a chess king
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            x2 = pos[0] + dx
            y2 = pos[1] + dy
            if x2 < 0 or x2 > 7 or y2 < 0 or y2 > 7:
                continue
            n.append((x2, y2))
            #time.sleep(2)
        return n

    def move_cost(self, a, b):
        for barrier in self.barriers:
            if b in barrier:
                return 100  # Extremely high cost to enter barrier squares
        return 1  # Normal movement cost


def AStarSearch(start, end, graph):
    G = {}  # Actual movement cost to each position from the start position
    F = {}  # Estimated movement cost of start to end going via this position

    # Initialize starting values
    G[start] = 0
    F[start] = graph.heuristic(start, end)

    closedVertices = set()
    openVertices = set([start])  # {(0,0)}
    cameFrom = {}
    currentFscore = 100
    while len(openVertices) > 0:
        #print(openVertices)
        # Get the vertex in the open list with the lowest F score
        current = None

        # TODO:--------------------------------------------
        # Loop through each position in the openVertices list and set the current
        # to be the position with the lowest F score, and put its F score in the
        # currentFscore variable.
        #
        # shuffle the order:
        # (A) currentFscore = F[pos]
        # (B) current = pos
        # (C) if current is None or F[pos] < currentFscore:
        # (D) for pos in openVertices:
        #print(currentFscore[0])
        for pos in openVertices:
            current = pos
            print(pos)
            print(current)
            print(F[pos])
            if current is None or F[pos] < currentFscore:
                currentFscore = F[pos]
                print(currentFscore)
        # -------------------------------------------------

        # Check if we have reached the goal
        if current == end:
            # Retrace our route backward
            path = [current]
            while current in cameFrom:
                current = cameFrom[current]
                path.append(current)
            path.reverse()
            return path, F[end]  # Done!

        # TODO:--------------------------------------------
        # Mark the currente vertx das closed, an remove it from openVertices.
        # you may be using: openVertices, closedVertices, remove(), add(), current
        # print([current])
        openVertices.remove(current)
        closedVertices.add(current)

        # -------------------------------------------------

        # Update scores for vertices near the current position
        for neighbour in graph.get_vertex_neighbours(current):
            if neighbour in closedVertices:
                continue  # We have already processed this node exhaustively
            candidateG = G[current] + graph.move_cost(current, neighbour)

            if neighbour not in openVertices:
                openVertices.add(neighbour)  # Discovered a new vertex
            elif candidateG >= G[neighbour]:
                continue  # This G score is worse than previously found

            cameFrom[neighbour] = current

            # TODO:--------------------------------------------
            # Adopt this G score
            G[neighbour] = candidateG
            H = graph.heuristic(neighbour, end)  # graph.heuristic(neighbour, start) or graph.heuristic(neighbour, end)
            F[neighbour] = G[neighbour] + H

    # -------------------------------------------------

    raise RuntimeError("A* failed to find a solution")


def draw_result(result_path, cost):
    plt.plot([v[0] for v in result_path], [v[1] for v in result_path])
    plt.plot([v[0] for v in result_path], [v[1] for v in result_path], 'o', color='lightblue')
    plt.plot([0, 7], [0, 7], 'o', color='red')
    for barrier in graph.barriers:
        plt.plot([v[0] for v in barrier], [v[1] for v in barrier], linewidth=20, color='yellow')
    plt.xlim(-1, 8)
    plt.ylim(-1, 8)
    plt.show()


def write_result(result_path, cost):
    print('Find the shortest path to get from (0,0) to (7,7):')
    print("Optimal Route = ", result_path)
    print("Optimal Cost =  ", cost)


if __name__ == "__main__":
    graph = AStarGraph()
    result_path, cost = AStarSearch((0, 0), (7, 7), graph)
    write_result(result_path, cost)
    draw_result(result_path, cost)
