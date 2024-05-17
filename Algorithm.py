from FileHandler import FileHandler
from GUI import GUI

import heapq
import time
from enum import Enum

import networkx as nx
import copy

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


class Heuristic(str, Enum):
    MANHATTAN = "Manhattan"
    EUCLIDIAN = "Euclidian"
    CHEBYSHEV = "Chebyshev"
    DIJKSTRA = "Dijkstra"


HEURISTICS = [e.value for e in Heuristic]


class Algorithm:
    """
    Algorithm of A*, it generates the shortest path for the given instance and shows the result using a GUI
    """

    def __init__(
        self,
        instance,
        logger,
    ):
        fh = FileHandler(instance)
        (
            self.E,
            self.V,
            self.vertices,
            self.edges,
            self.edge_index,
            self.nodes,
        ) = fh.read()
        # E is total number of edges
        # V is total number of vertices/nodes
        # vertices is a dictionary that contain vertex id as key and a tuple (position x, position y) as value
        # edges is a list of tuples (source, destination, weight, color)
        # edge_index is a dictionary that stores the index of each edges in the list above
        # nodes is a dictionary where we can get direct neighbors of each nodes and the color assigned to each node
        logger.debug("Instance loaded")
        logger.debug(f"{self.V} nodes = {self.vertices}")
        logger.debug(f"{self.E} edges = {self.edges}")
        logger.debug(f"Edges index = {self.edge_index}")
        logger.debug(f"Nodes neighbors = {self.nodes}")

        # Creating graph object
        self.G = nx.Graph()
        edges_mod = [
            (x[0], x[1], {"weight": x[2], "color": x[3]}) for x in self.edges
        ]
        self.G.add_edges_from(edges_mod)

        # These attributes are used for the graphical aspect of the code.
        self.history = []

        # Path and cost of the best solution found
        self.path = []
        self.cost = 0
        self.logger = logger
        self.name = "Q-learning"
        

    def show(self):
        """
        Launch GUI
        """
        gui = GUI(
            self.history,
            self.G,
            self.vertices,
            self.name,
            self.logger,
        )
        gui.show()

    def run(self):
        """
        Run desired solve function with time measurement
        """
        self.logger.info(self.name)
        start = time.time()

        self.q_learn()

        end = time.time()
        self.logger.info(f"Time elapsed: {(end-start)*1000:.2f}ms")

    def q_learn(self, alpha = 0.1, gamma = 0.9, epsilon = 0.1, epsilon_min=0.01, epsilon_decay=0.995, episodes = 3000):
        
        eps = epsilon

        """Creating Q matrix"""
        Q = np.matrix(np.zeros(shape=(self.V,self.V)))

        """Creating R matrix"""
        R = np.matrix(np.zeros(shape=(self.V,self.V)))
        for x in range (self.V):
            for y in range (self.V):
                R[x,y] = -math.dist(self.vertices[x],self.vertices[y])

        """Metrics for improvement tracking"""
        tour_lenghts = []

        for episode in range(episodes):
            start = np.random.randint(0,self.V)
            state = start
            visited = set()
            visited.add(state)

            tour_lenght = 0

            while len(visited) < self.V:
                # ε-greedy policy
                if np.random.uniform(0,1) < epsilon:
                    # Explore: go to a random node
                    next_state = np.random.randint(0,self.V)
                else:
                    # Exploit: go to the best node (the closest one)
                    possible_next_states = Q[state,:]
                    masked_states = np.ma.array(possible_next_states, mask=[i in visited for i in range(self.V)])
                    next_state = masked_states.argmax()

                if next_state not in visited:
                    reward = R[state,next_state]
                    Q[state,next_state] = Q[state,next_state] + alpha * (reward + gamma * np.max(Q[next_state,:]) - Q[state,next_state])
                    tour_lenght -= reward

                    state = next_state
                    visited.add(state)

            reward = R[state, start]
            Q[state,start] = Q[state,start] + alpha * (reward + gamma * np.max(Q[start,:]) - Q[state,start])
            tour_lenght -= reward

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            tour_lenghts.append(tour_lenght)
            best
            if tour_lenght < best:
                best = tour_lenght

            # Print the progress every few episodes
            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode + 1}/{episodes} completed")
                
            tour = self.extract_tour(Q, start)
            self.history.append((tour_lenght, tour))

            
        """Extract the tour"""
        tour = self.extract_tour(Q, start)
        self.plot_evolution(tour_lenghts, {"alpha": alpha, "gamma": gamma, "epsilon": eps, "epsilon_min": epsilon_min, "epsilon_decay": epsilon_decay})
        print("Tour: ", tour)

    def extract_tour(self, Q_original, start):
        Q = copy.deepcopy(Q_original)
        current_node = start
        tour = [start]
        while len(tour) < self.V:
            next_node = np.argmax(Q[current_node,])
            while next_node in tour:
                Q[current_node,next_node] = -np.inf # Prevent from going back to the same node
                next_node = np.argmax(Q[current_node,])
            tour.append(next_node)
            current_node = next_node
        tour.append(start)
        return tour
    
    def plot_evolution(self, tour_lenghts, parameters):
        window_size = 100
        moving_avg = np.convolve(tour_lenghts, np.ones(window_size)/window_size, mode='valid')

        plt.figure(figsize=(8,5))
        plt.plot(tour_lenghts)
        plt.title('alpha = {}, gamma = {}, epsilon = {}, epsilon_min = {}, epsilon_decay = {}'.format(parameters["alpha"], parameters["gamma"], parameters["epsilon"], parameters["epsilon_min"], parameters["epsilon_decay"]))
        plt.ylabel('Tour lenght')
        plt.xlabel('Episode')
        plt.show()
