import matplotlib.pyplot as plt
import pickle as pkl


def plot_learning_curve(value_dict, xlabel='step'):
    # Plot step vs the mean(last 50 episodes' rewards)
    fig = plt.figure(figsize=(12, 4 * len(value_dict)))

    for i, (key, values) in enumerate(value_dict[0].items()):
        ax = fig.add_subplot(len(value_dict[0]), 1, i + 1)
        ax.plot(range(len(values))[-1000:], values[-1000:])
        ax.plot(range(len(value_dict[1][key]))[-1000:], value_dict[1][key][-1000:])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(key)
        ax.grid('k--', alpha=0.6)

    plt.tight_layout()

    plt.savefig('plotdqn' + ".png")


with open('my_dict_(-r).pkl', 'rb') as f:
    dict1 = pkl.load(f)
with open('my_dict_r.pkl', 'rb') as f:
    dict2 = pkl.load(f)

dict = [dict1,dict2]
plot_learning_curve(dict)
