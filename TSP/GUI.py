import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

matplotlib.use("TkAgg")



class GUI:
    """
    Class used for the GUI of the project
    """

    def __init__(
        self, history, graph, nodes_pos, name, logger
    ):
        self.history = history
        self.G = graph
        self.nodes_pos = nodes_pos
        self.nodes_pos_without_goals = b = {
            x: self.nodes_pos[x] for x in self.nodes_pos if x not in [0, 1]
        }
        self.name = name
        self.logger = logger
        self.index = 0

        self.edges_index = list(self.G.edges())
        self.node_labels = {x: "{}".format(x) for x in list(self.G.nodes())}

    def plotOneState(self, state, iteration):

        # revstate is filled only when bidirectionnal search is performed
        print(state)
        # Set title, texts and legends
        plt.clf()
        plt.cla()
        ax = plt.gca()
        ax.set_title(f"{self.name}: iteration {iteration+1}")
        # Compute current path cost
        (score, path) = state
        path_cost = score
        final = path[-1] == 1 and path[0] == 0

        plt.gcf().text(
            0.4,
            0.15,
            f" Path cost = {path_cost:.2f}",
            fontsize=11,
        )


        # Fill nodes and edges color wrt current state
        edges_color = ["black" for _ in self.G.edges()]
        nodes_color = ["grey" for _ in self.G.nodes()]

        
        # current
        for current_state in state[0:]:
            list_edges = []
            for i, k in zip(path[0::1], path[1::1]):
                list_edges.append([i, k])
           
        print(list_edges)
        # Draw all nodes and edges
        nx.draw_networkx_nodes(
            self.G,
            self.nodes_pos,
            nodelist=self.nodes_pos,
            node_color=nodes_color,
            node_size=400,
        )
        nx.draw_networkx_labels(
            self.G,
            self.nodes_pos,
            labels=self.node_labels,
            font_size=10
        )
        nx.draw_networkx_edges(
            self.G,
            self.nodes_pos,
            edgelist=list_edges,
            edge_color=edges_color
    )

        plt.subplots_adjust(bottom=0.2)

    def show(self):
        self.showButtons()

    def plotIndex(self):
        self.plotOneState(self.history[self.index], iteration=self.index)

    def showButtons(self):
        def next(event):
            self.index += 1
            self.index = min(self.index, len(self.history) - 1)
            self.showButtons()

        def back(event):
            self.index -= 1
            self.index = max(self.index, 0)
            self.showButtons()

        def end(event):
            self.index = len(self.history) - 1
            self.showButtons()

        def reset(event):
            self.index = 0
            self.showButtons()

        self.plotIndex()
        self.button_reset = widgets.Button(
            plt.axes([0.5, 0.01, 0.1, 0.05]), "Reset"
        )
        self.button_back = widgets.Button(
            plt.axes([0.6, 0.01, 0.1, 0.05]), "Back"
        )
        self.button_next = widgets.Button(
            plt.axes([0.7, 0.01, 0.1, 0.05]), "Next"
        )
        self.button_end = widgets.Button(
            plt.axes([0.8, 0.01, 0.1, 0.05]), "End"
        )

        self.button_next.on_clicked(next)
        self.button_back.on_clicked(back)
        self.button_end.on_clicked(end)
        self.button_reset.on_clicked(reset)
        plt.show()
