import matplotlib.pyplot as plt

def plot_trajectory(x,
                    y,
                    title,
                    save_path=None):

    plt.figure(figsize=(8,6))

    plt.plot(x, y)

    plt.title(title)

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.close()