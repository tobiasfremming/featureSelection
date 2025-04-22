import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
import umap


def load_data(path: str, fitness_column: str):
    # Load CSV with binary feature strings as X data and model error as y data
    lookup_table = pd.read_csv(path, dtype={"key": str})#, header=None, names=["features_binary", "error", "pen", "f32", "f16", "f8"], dtype={"features_binary": str})
    lookup_table = lookup_table.tail(-1).reset_index(drop=True)  # delete the first row (no features and error of 1 will distort the plot)
    # Convert binary strings to feature vectors
    lookup_table['features'] = lookup_table['key'].apply(lambda x: [int(bit) for bit in x])
    # print(lookup_table['features'].head)
    X = np.stack(lookup_table['features'].values)
    y = lookup_table[fitness_column].values
    return lookup_table, X, y

test_lut, X, y = load_data("Lookup_tables/diabetes_fitness_lut.csv", 'err')
print(test_lut)


def plot_fitness_landscape_TSNE_4D(lookup_table_path: str):
    """
    Visualizes a fitness landscape from a lookup table reduced to 3 dims with the error colored as 4th dim
    """
    lut, X, y = load_data(lookup_table_path)
    
    # Apply t-SNE to reduce to 3 dimensions
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, init='pca')
    X_tsne = tsne.fit_transform(X)

    # Plot in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Use color to indicate fitness (lower = better)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y, cmap='viridis')
    fig.colorbar(scatter, ax=ax, label='Error (Fitness)')

    # Labels
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.set_zlabel('t-SNE Dim 3')
    ax.set_title('Fitness Landscape using t-SNE')

    plt.tight_layout()
    plt.show()


def plot_fitness_landscape_PCA_3D(lookup_table_path: str):
    # Load CSV with binary feature strings and error
    lut, X, y = load_data(lookup_table_path)

    # Apply PCA to reduce 8D feature space to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Extract X, Y for grid interpolation
    x_vals = X_pca[:, 0]
    y_vals = X_pca[:, 1]
    z_vals = y  # Error values

    # Create a regular grid to interpolate onto
    grid_x, grid_y = np.mgrid[
        x_vals.min():x_vals.max():100j,  # 100 points in x-direction
        y_vals.min():y_vals.max():100j   # 100 points in y-direction
    ]

    # Interpolate the scattered data onto the grid
    grid_z = griddata((x_vals, y_vals), z_vals, (grid_x, grid_y), method='cubic')

    # Plot the fitness landscape as a surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, label='Model Error (Fitness)')

    ax.set_xlabel('PCA Dim 1')
    ax.set_ylabel('PCA Dim 2')
    ax.set_zlabel('Error (Fitness)')
    ax.set_title('Fitness Landscape (PCA + Surface Plot)')

    plt.tight_layout()
    plt.show()


def plot_fitness_landscape_3D(path: str, 
                              method: str = 'pca', 
                              show_points: bool = False, 
                              highlight_genotypes: list[str] = None,
                              f_column: str = 'f16'):
    """
    Visualizes the fitness landscape.

    Parameters:
        path (str): path to the lookup table difining the fitness landscape. 
                    -its expected to have 2 columns, the first one representing the features,
                     the second one representing the error or fitness of the individual
        method (str): Dimensionality reduction method: 'pca', 'tsne', or 'umap'
        show_points (bool): Whether to overlay original points on the surface
    """

    # Load the lookup table data
    lut, X, y = load_data(path, f_column)

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, init='pca', perplexity=30)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method. Choose from 'pca', 'tsne', or 'umap'.")

    X_reduced = reducer.fit_transform(X)
    x_vals, y_vals = X_reduced[:, 0], X_reduced[:, 1]
    z_vals = y
    ###
    min_value = np.min(z_vals)
    min_index = np.argmin(z_vals)
    # print(min_value)
    # print(min_index)

    # Interpolation onto a grid
    grid_x, grid_y = np.mgrid[
        x_vals.min():x_vals.max():100j,
        y_vals.min():y_vals.max():100j
    ]
    # grid_z = griddata((x_vals, y_vals), z_vals, (grid_x, grid_y), method='cubic')
    grid_z = griddata((x_vals, y_vals), z_vals, (grid_x, grid_y), method='linear')  
    # grid_z = griddata((x_vals, y_vals), z_vals, (grid_x, grid_y), method='nearest')


    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    surf = ax.plot_surface(grid_x, grid_y, grid_z,
                       cmap='viridis', edgecolor='none',
                       alpha=0.8, zorder=0)
    fig.colorbar(surf, ax=ax, label='Model Error (Fitness)')

    # Overlay original points
    # if show_points:
    #     ax.scatter(x_vals, y_vals, z_vals, color='black', s=10, label='Feature subset')
    #     ax.legend()
    if show_points:
        ax.scatter(x_vals, y_vals, z_vals,
                color='black', s=10, alpha=0.4,
                label='Feature subset', zorder=1)

    # Highlight specific genotypes
    if highlight_genotypes:
        cmap = plt.cm.viridis
        n = len(highlight_genotypes)
        for i, genotype in enumerate(highlight_genotypes):
            if genotype not in lut['key'].values:
                print(f"Warning: Genotype '{genotype}' not in lookup table.")
                continue
            index = lut[lut['key'] == genotype].index[0]
            # print(index)
            # point = X[index].reshape(1, -1)
            # reduced_point = reducer.transform(point)
            # reduced_point = X_reduced[index].reshape(1, -1)
            x_point = x_vals[index]
            y_point = y_vals[index]
            z_point = z_vals[index]
            # print(x_point, y_point, z_point)
            color = cmap(i / max(n - 1, 1))  # evenly spaced colors
            ax.scatter(x_point, y_point, z_point,
                       color=color, s=100, edgecolor='black', facecolor=color, linewidth=1.2,
                       marker='o', zorder=200)


    ax.set_xlabel(f'{method.upper()} Dimension 1')
    ax.set_ylabel(f'{method.upper()} Dimension 2')
    ax.set_zlabel('Model Error')
    ax.set_title(f'Fitness Landscape ({method.upper()})')

    plt.tight_layout()
    plt.show()


def plot_fitness_landscape_4D(path: str, method: str = 'pca'):
    """
    Visualizes the fitness landscape.

    Parameters:
        path (str): path to the lookup table difining the fitness landscape. 
                    -its expected to have 2 columns, the first one representing the features,
                     the second one representing the error or fitness of the individual
        method (str): Dimensionality reduction method: 'pca', 'tsne', or 'umap'
    """

    # Load the lookup table data
    lut, X, y = load_data(path, 'err')

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=3)
    elif method == 'tsne':
        reducer = TSNE(n_components=3, random_state=42, init='pca', perplexity=30)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=3, random_state=42)
    else:
        raise ValueError("Invalid method. Choose from 'pca', 'tsne', or 'umap'.")

    X_reduced = reducer.fit_transform(X)
    x_vals, y_vals, z_vals = X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2]

    # Plot in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Use color to indicate fitness (lower = better)
    scatter = ax.scatter(x_vals, y_vals, z_vals, c=y, cmap='viridis')
    fig.colorbar(scatter, ax=ax, label='Error (Fitness)')

    # Labels
    ax.set_xlabel(f'{method.upper()} Dimension 1')
    ax.set_ylabel(f'{method.upper()} Dimension 2')
    ax.set_zlabel(f'{method.upper()} Dimension 3')
    ax.set_title(f'Fitness Landscape ({method.upper()})')

    plt.tight_layout()
    plt.show()


def find_local_minima(path: str, fitness_col="err"):
    bit_col = "key"
    lookup_table = pd.read_csv(path, dtype={"key": str})
    lookup_table = lookup_table.tail(-1)
    minima = []

    # Make a dictionary for fast lookup
    fitness_dict = dict(zip(lookup_table[bit_col], lookup_table[fitness_col]))

    for bitstring, fitness in fitness_dict.items():
        n_bits = len(bitstring)
        is_local_min = True
        # print("#########")
        # print(bitstring, fitness)

        # Generate all neighbors
        for i in range(n_bits):
            neighbor = list(bitstring)
            neighbor[i] = '1' if bitstring[i] == '0' else '0'
            neighbor_str = ''.join(neighbor)
            # print(neighbor_str)

            # Check if neighbor has better or equal fitness
            if neighbor_str in fitness_dict and fitness_dict[neighbor_str] < fitness:
                # print(fitness_dict[neighbor_str])
                is_local_min = False
                break

        if is_local_min:
            minima.append((bitstring, fitness))

    return minima


def find_global_minimum(path: str, fitness_col="err"):
    bit_col = "key"
    lookup_table = pd.read_csv(path, dtype={"key": str})
    lookup_table = lookup_table.tail(-1)
    # Locate the row with the minimum fitness
    min_row = lookup_table.loc[lookup_table[fitness_col].idxmin()]
    genotype = min_row[bit_col]
    fitness = min_row[fitness_col]
    return genotype, fitness


lut_cancer = "Lookup_tables/cancer_fitness_lut.csv"
lut_diabetes = "Lookup_tables/diabetes_fitness_lut.csv"
lut_heart = "Lookup_tables/heart_fitness_lut.csv"
lut_wine = "Lookup_tables/wine_fitness_lut.csv"
f_value = "f8"
lut = lut_heart


local_minima = find_local_minima(lut, f_value)
len(local_minima)
local_minima_list = [genotype for genotype, _ in local_minima]
local_minima_list
global_minimum = find_global_minimum(lut, f_value)
global_minimum
global_minimum_genotype = global_minimum[0]
global_minimum_genotype
# actual global minimum:
# lut = pd.read_csv(lut_wine)
global_min = lut['err'].min()
global_min

# plot_fitness_landscape_TSNE_4D(lookup_table_path)
# plot_fitness_landscape_PCA_3D(lookup_table_path)
plot_fitness_landscape_3D(lut, 'pca', True, local_minima_list, f_column=f_value)
plot_fitness_landscape_3D(lut, 'tsne', True, local_minima_list, f_column=f_value)
plot_fitness_landscape_3D(lut, 'umap', True, local_minima_list, f_column=f_value)
plot_fitness_landscape_4D(lut, 'pca')
plot_fitness_landscape_4D(lookup_table_path, 'tsne')
plot_fitness_landscape_4D(lookup_table_path, 'umap')
