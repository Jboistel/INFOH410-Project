import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

matplotlib.use("TkAgg")

class GUI:
    """
    Class used for the GUI of the project
    """

    def __init__(self, history, graph, nodes_pos, name, logger, best_index):
        self.history = history
        self.G = graph
        self.nodes_pos = nodes_pos
        self.nodes_pos_without_goals = {
            x: self.nodes_pos[x] for x in self.nodes_pos if x not in [0, 1]
        }
        self.best_index = best_index
        self.name = name
        self.logger = logger
        self.index = 0

        self.edges_index = list(self.G.edges())
        self.node_labels = {x: "{}".format(x) for x in list(self.G.nodes())}

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)  # Adjust to make space for the slider and buttons

        self.ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = widgets.Slider(
            ax=self.ax_slider,
            label="Episode",
            valmin=0,
            valmax=len(self.history) - 1,
            valinit=self.index,
            valfmt='%0.0f'
        )
        self.slider.on_changed(self.slide)

        self.ax_reset = plt.axes([0.8, 0.01, 0.1, 0.05])
        self.button_reset = widgets.Button(self.ax_reset, "Reset")
        self.button_reset.on_clicked(self.reset)

        self.ax_back = plt.axes([0.7, 0.01, 0.1, 0.05])
        self.button_back = widgets.Button(self.ax_back, "Back")
        self.button_back.on_clicked(self.back)

        self.ax_next = plt.axes([0.6, 0.01, 0.1, 0.05])
        self.button_next = widgets.Button(self.ax_next, "Next")
        self.button_next.on_clicked(self.next)

        self.ax_end = plt.axes([0.5, 0.01, 0.1, 0.05])
        self.button_end = widgets.Button(self.ax_end, "End")
        self.button_end.on_clicked(self.end)

        self.ax_best = plt.axes([0.4, 0.01, 0.1, 0.05])
        self.button_best = widgets.Button(self.ax_best, "Best")
        self.button_best.on_clicked(self.best)

        self.plotIndex()

    def plotOneState(self, state, iteration):
        # Set title, texts and legends
        self.ax.clear()
        self.ax.set_title(f"{self.name}: iteration {iteration + 1}")

        # Compute current path cost
        (score, path) = state
        path_cost = score

        # Remove existing text objects before adding new text
        for txt in self.ax.texts:
            txt.set_visible(False)

        self.ax.text(
            0.4,
            -0.1,
            f"Path cost = {path_cost:.2f}",
            fontsize=11,
            transform=self.ax.transAxes
        )

        # Fill nodes and edges color wrt current state
        edges_color = ["black" for _ in self.G.edges()]
        nodes_color = ["grey" for _ in self.G.nodes()]

        # current
        list_edges = []
        for i, k in zip(path[0::1], path[1::1]):
            list_edges.append([i, k])

        # Draw all nodes and edges
        nx.draw_networkx_nodes(
            self.G,
            self.nodes_pos,
            nodelist=self.nodes_pos,
            node_color=nodes_color,
            node_size=400,
            ax=self.ax
        )
        nx.draw_networkx_labels(
            self.G,
            self.nodes_pos,
            labels=self.node_labels,
            font_size=10,
            ax=self.ax
        )
        nx.draw_networkx_edges(
            self.G,
            self.nodes_pos,
            edgelist=list_edges,
            edge_color=edges_color,
            ax=self.ax
        )

        self.fig.canvas.draw_idle()

    def plotIndex(self):
        self.plotOneState(self.history[self.index], iteration=self.index)

    def next(self, event):
        self.index += 1
        self.index = min(self.index, len(self.history) - 1)
        self.update_slide()
        self.plotIndex()

    def back(self, event):
        self.index -= 1
        self.index = max(self.index, 0)
        self.update_slide()
        self.plotIndex()

    def end(self, event):
        self.index = len(self.history) - 1
        self.update_slide()
        self.plotIndex()

    def reset(self, event):
        self.index = 0
        self.update_slide()
        self.plotIndex()
    
    def best(self, event):
        self.index = self.best_index
        self.update_slide()
        self.plotIndex()

    def slide(self, val):
        try:
            int_val = int(val)
            if 0 <= int_val < len(self.history):
                self.index = int_val
                self.plotIndex()
            else:
                print("Slider value out of range")
        except ValueError:
            print("Invalid slider value")

    def update_slide(self):
        self.slider.set_val(self.index)
        self.plotIndex()

    def show(self):
        plt.show()

# Example usage:
# history = [(1, [0, 2, 3, 1]), (2, [0, 3, 1]), (1.5, [0, 4, 1])]
# G = nx.path_graph(5)
# pos = nx.spring_layout(G)
# gui = GUI(history, G, pos, "Example", None)
# gui.show()
