import numpy as np
import matplotlib.pyplot as plt
import dill as pickle

class Ghostleg:
    def __init__(self, num_layers, num_possible_bridges, bridge_density):
        self.num_possible_bridges = num_possible_bridges
        self.bridge_density = bridge_density
        self.num_layers = num_layers
        self.num_bridges = self.num_layers - 1
        self.bridges = np.zeros((self.num_bridges, self.num_possible_bridges))

        for i in range(self.num_bridges):
            for j in range(self.num_possible_bridges):
                num_neighbors = 0
                if self.bridges[(self.num_bridges + i - 1) % self.num_bridges][j] == 1 or self.bridges[(i + 1) % (self.num_bridges - 1)][j] == 1:
                    num_neighbors += 1
                if np.random.random() < self.bridge_density and num_neighbors == 0:
                    self.bridges[i][j] = 1

    def drop(self, index):
        temp = index
        indices = np.zeros(self.num_possible_bridges)
        indices[0] = index
        for i in range(len(self.bridges[index])):
            if self.bridges[index][i] == 1:
                temp += 1
                temp %= self.num_layers
            elif self.bridges[index][(self.num_layers + i - 1) % self.num_layers] == 1:
                temp -= 1
                temp += self.num_layers
                temp %= self.num_layers
            if temp < 0 or temp >= self.num_layers:
                raise Exception("Ghostleg is broken")
            if i < self.num_possible_bridges:
                indices[i] = temp
        indices = [int(i) for i in indices]
        return indices

    def visualize(self):
        bridge_plot = plt.imshow(self.bridges, cmap='binary')
        # plt.imshow(self.layers, cmap='binary')
        plt.show()

    def drop_visualize(self, index):
        path = self.drop(index)
        layer_range = range(self.num_possible_bridges)
        plt.plot(path, layer_range)
        plt.xlim(0, self.num_layers)
        plt.ylim(0, self.num_possible_bridges)
        plt.show()
        print(path[-1])

    def drop_visualize_all(self, indices):
        layer_range = range(self.num_possible_bridges)
        paths = []
        for i in indices:
            paths.append(self.drop(i))
            plt.plot(paths[-1], layer_range)
        plt.xlim(0, self.num_layers)
        plt.ylim(0, self.num_possible_bridges)
        plt.show()
        final_paths = [paths[i][-1] for i in range(len(paths))]
        print(f"first drops = {indices}")
        print(f"{final_paths = }")

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


def main():
    ghostleg = Ghostleg(10, 25, 0.5)
    # ghostleg.visualize()
    # ghostleg.drop_visualize(np.random.randint(0, ghostleg.num_layers))
    test_drops = [np.random.randint(0, ghostleg.num_layers) for i in range(10)]
    ghostleg.drop_visualize_all(test_drops)
    ghostleg.save('ghostleg.pkl')
    ghostleg = ghostleg.load('ghostleg.pkl')
    ghostleg.drop_visualize_all(test_drops)

if __name__ == "__main__":
    main()
