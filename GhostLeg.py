import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
import click


class Ghostleg:
    def __init__(self, num_layers, num_possible_bridges, bridge_density):
        if num_layers < 2:
            raise Exception("Number of layers must be at least 2")
        if num_possible_bridges < 2:
            raise Exception("Number of possible bridges must be at least 2")
        if bridge_density < 0 or bridge_density > 1:
            raise Exception("Bridge density must be between 0 and 1")
        self.num_possible_bridges = num_possible_bridges
        self.bridge_density = bridge_density
        self.num_layers = num_layers
        self.num_bridges = self.num_layers - 1
        self.bridges = np.zeros((self.num_bridges, self.num_possible_bridges))

        for i in range(self.num_bridges):
            for j in range(self.num_possible_bridges):
                num_neighbors = 0
                if self.bridges[(self.num_bridges + i - 1) % self.num_bridges][j] == 1 or \
                        self.bridges[(i + 1) % (self.num_bridges - 1)][j] == 1:
                    num_neighbors += 1
                if np.random.random() < self.bridge_density and num_neighbors == 0:
                    self.bridges[i][j] = 1

    def drop(self, index):
        if index not in range(self.num_layers):
            raise Exception("Index out of range")
        temp = index
        indices = np.zeros(self.num_possible_bridges)
        indices[0] = index
        for i in range(len(self.bridges[indices[0]])):
            if self.bridges[indices[i] - 1][i] == 1:
                temp += 1
                temp %= self.num_layers
            elif self.bridges[indices[i] - 1][(self.num_layers + i - 1) % self.num_layers] == 1:
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
        layer_range = range(self.num_possible_bridges)
        if index not in range(self.num_layers):
            raise Exception("Index out of range")
        path = self.drop(index)
        plt.plot(path, layer_range)
        plt.xlim(0, self.num_layers)
        plt.ylim(0, self.num_possible_bridges)
        plt.show()
        print(path[-1])

    def drop_visualize_all(self, indices):
        layer_range = range(self.num_possible_bridges)
        paths = []
        for i in indices:
            if i not in range(self.num_layers):
                raise Exception("Index out of range")
            paths.append(self.drop(i))
            plt.plot(paths[-1], layer_range)
        plt.xlim(0, self.num_layers)
        plt.ylim(0, self.num_possible_bridges)
        plt.show()
        final_paths = [paths[i][-1] for i in range(len(paths))]
        print(f"first drops = {indices}")
        print(f"{final_paths = }")

    def drop_all(self, indices):
        if len(indices) == 0:
            raise Exception("Indices must be a non-empty list")
        print(f"first drops = {indices}")
        for i in indices:
            if i not in range(self.num_layers):
                raise Exception("Index out of range")
        return [self.drop(i) for i in indices]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


@click.command()
@click.option('--num_layers', default=10, help='Number of layers in the ghostleg')
@click.option('--num_possible_bridges', default=25, help='Number of possible bridges in the ghostleg')
@click.option('--bridge_density', default=0.5, help='Density of bridges in the ghostleg')
@click.option('--num_drops', default=0, help='Number of drops to make')
@click.option('--filename', default='ghostleg.pkl', help='Filename to save or load the ghostleg to')
@click.option('--load', is_flag=True, default=False, help='Load a ghostleg from a file')
@click.option('--save', is_flag=True, help='Save the ghostleg to a file', default=False)
@click.option('--visualize', is_flag=True, help='Visualize the ghostleg', default=False)
@click.option('--indices', default=[1, 2, 3], help='Indices of the drops to visualize', type=str)
def main(num_layers, num_possible_bridges, bridge_density, num_drops, filename, load, save, visualize, indices):
    if type(num_layers) != int or type(num_possible_bridges) != int or type(bridge_density) != float:
        raise Exception("Invalid input type")
    if num_layers <= 0 or num_possible_bridges <= 0 or bridge_density < 0 or bridge_density > 1:
        raise Exception("Invalid input value")
    if load and save:
        raise Exception("Cannot load and save at the same time")
    if not filename and (load or save):
        raise Exception('Filname must be specified when loading or saving')
    if type(indices) != str:
        raise Exception("Invalid input type : indices")
    print(f"first drops = {indices}")
    if load:
        ghostleg = Ghostleg.load(filename)
    else:
        ghostleg = Ghostleg(num_layers, num_possible_bridges, bridge_density)
    if visualize: ghostleg.visualize()
    if indices == [0] and num_drops == 0:
        ghostleg.drop(np.random.randint(0, ghostleg.num_layers - 1))
    elif indices is [0]:
        indices = [i for i in range(num_layers)]
    elif type(indices) == str:
        try:
            indices.remove('['); indices.remove(']')
        except:
            pass
        indices = indices.split(',')
        indices = [int(i) for i in indices]
    elif num_drops == 0:
        indices = [i for i in range(num_layers)]
    else:
        indices = [i for i in range(num_layers)]
    ghostleg.drop_all(indices)
    if save and not filename.endswith('.pkl'):
        filename += '.pkl'
    if save: ghostleg.save(filename)


if __name__ == "__main__":
    main()
