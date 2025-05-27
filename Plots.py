import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

#ALOT OF CONTENT IN THIS FILE HAS BEEN GENERATED USING PARTLY OR ONLY CHATGPT,
#I am not great at creating plots, and hope it's ok.

class Action(Enum):
    Stand = 1
    Hit = 0

def index_to_state(index):
    #CHATGPT
    has_soft_ace = index % 2
    index //= 2
    dealer_card = (index % 11) + 1
    index //= 11
    player_total = index + 1
    return (player_total, dealer_card, bool(has_soft_ace))

def visualize_win_rates(q_table, epochs, results, soft_ace=True):
    #CHATGPT
    """
    Visualize estimated win rates from Q-table for states with or without soft ace.
    Win rates are approximated as max Q-value per state.
    """
    soft_ace_flag = 1 if soft_ace else 0

    win_rate_matrix = np.zeros((32, 10))

    for player_total in range(1, 33):
        for dealer_card in range(1, 11):
            idx = ((player_total - 1) * 11 + (dealer_card - 1)) * 2 + soft_ace_flag

            # Take max Q-value as proxy for win rate, clip between 0 and 1 for visualization
            max_q = np.max(q_table[idx])
            win_rate = np.clip(max_q, 0, 1)  # assuming Q-values roughly in [0,1]
            win_rate_matrix[player_total - 1, dealer_card - 1] = win_rate

    plt.figure(figsize=(14, 10))
    sns.heatmap(win_rate_matrix, annot=np.round(win_rate_matrix, 2), fmt='.2f',
                cmap='YlGnBu', xticklabels=range(2, 12), yticklabels=range(0, 32))
    plt.title(f"Estimated Win Rates (Soft Ace = {soft_ace})")
    plt.xlabel("Dealer Face-up Card")
    plt.ylabel("Player Total")

    # Display the results string outside the plot area
    plt.figtext(0.15, 0.02, f"Epochs: {epochs}\n{results}", fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    # Save the plot as an image file with the filename based on epochs
    filename = f"{epochs}.png"
    plt.savefig(filename, bbox_inches='tight')

    plt.show()


#the method I gave chatGPT.
def plot_tracking_1(win_track, lose_track, draw_track, number_of_games):
    x = np.arange(number_of_games)

    plt.plot(x, win_track, color='g', label='total wins')
    plt.plot(x, lose_track, color='r', label='total loses')
    plt.plot(x, draw_track, color='y', label='total draws')

    plt.legend()
    plt.show()

#created the one above, asked chatGPT to include win_rate, lose_rate and draw_rate and also store it in a file.
#also asked it to include the value of the hyperparameters, makes it easier to evaluate and discuss in paper.
def plot_tracking(win_track, lose_track, draw_track, number_of_games,
                  epsilon, discount_factor, tuning_count,
                  epsilon_min=None, epsilon_decay=None,
                  learning_rate=None, learning_rate_decay=None, learning_rate_min=None):

    # Create output directory
    output_dir = "hyperparameter_training2"

    #x = np.arange(1, number_of_games + 1)
    x = np.logspace(0, np.log10(number_of_games), num=number_of_games, dtype=int)
    x = np.unique(x)  # remove duplicates caused by int rounding


    #win_cumulative = np.array(win_track)
    #lose_cumulative = np.array(lose_track)
    #draw_cumulative = np.array(draw_track)

    win_cumulative = np.array([win_track[i - 1] for i in x])
    lose_cumulative = np.array([lose_track[i - 1] for i in x])
    draw_cumulative = np.array([draw_track[i - 1] for i in x])


    win_rate = win_cumulative / x
    lose_rate = lose_cumulative / x
    draw_rate = draw_cumulative / x

    fig, axs = plt.subplots(2, 1, figsize=(12, 9))

    # --- Subplot 1: Total Counts ---
    axs[0].plot(x, win_cumulative, color='g', label='Total Wins')
    axs[0].plot(x, lose_cumulative, color='r', label='Total Losses')
    axs[0].plot(x, draw_cumulative, color='y', label='Total Draws')
    axs[0].set_title('Game Outcome Totals')
    axs[0].legend()
    axs[0].set_xlabel('Number of Games')
    axs[0].set_ylabel('Count')

    axs[0].set_xscale('log')
    axs[1].set_xscale('log')


    # Annotate final total values
    axs[0].annotate(f'{win_cumulative[-1]} wins', xy=(x[-1], win_cumulative[-1]),
                    xytext=(-60, 10), textcoords='offset points', color='g',
                    arrowprops=dict(arrowstyle='->', color='g'))

    axs[0].annotate(f'{lose_cumulative[-1]} losses', xy=(x[-1], lose_cumulative[-1]),
                    xytext=(-60, -10), textcoords='offset points', color='r',
                    arrowprops=dict(arrowstyle='->', color='r'))

    axs[0].annotate(f'{draw_cumulative[-1]} draws', xy=(x[-1], draw_cumulative[-1]),
                    xytext=(-60, 10), textcoords='offset points', color='y',
                    arrowprops=dict(arrowstyle='->', color='y'))

    # --- Subplot 2: Rates ---
    axs[1].plot(x, win_rate, color='g', label='Win Rate')
    axs[1].plot(x, lose_rate, color='r', label='Lose Rate')
    axs[1].plot(x, draw_rate, color='y', label='Draw Rate')
    axs[1].set_title('Game Outcome Rates')
    axs[1].legend()
    axs[1].set_xlabel('Number of Games')
    axs[1].set_ylabel('Rate')

    # Annotate final win rate
    final_win_rate = win_rate[-1]
    axs[1].annotate(f'{final_win_rate:.2%} win rate', xy=(x[-1], final_win_rate),
                    xytext=(-100, 10), textcoords='offset points', color='g',
                    arrowprops=dict(arrowstyle='->', color='g'))

    # --- Hyperparameters Box ---
    hyperparams_text = f"""Hyperparameters:
Epsilon: {epsilon}
Discount Factor: {discount_factor}"""
    if epsilon_min is not None:
        hyperparams_text += f"\nEpsilon Min: {epsilon_min}"
    if epsilon_decay is not None:
        hyperparams_text += f"\nEpsilon Decay: {epsilon_decay}"
    if learning_rate is not None:
        hyperparams_text += f"\nLearning Rate: {learning_rate}"
    if learning_rate_decay is not None:
        hyperparams_text += f"\nLR Decay: {learning_rate_decay}"
    if learning_rate_min is not None:
        hyperparams_text += f"\nLR Min: {learning_rate_min}"

    plt.gcf().text(0.75, 0.5, hyperparams_text, fontsize=10,
                   bbox=dict(facecolor='white', edgecolor='black'))

    plt.tight_layout(rect=[0, 0, 0.75, 1])

    # --- Filename Formatting ---
    def fmt(val): return str(val).replace('.', 'p') if val is not None else "NA"

    filename = (
        f"{output_dir}/hyperparameter_testing2_"
        f"{tuning_count}tuning_"
        f"{win_rate[-1]:.4f}_win_rate_.png"
    )

    plt.savefig(filename)
    plt.close()

    return win_rate[-1]

def _get_Action_String(actions):
    if actions[0]==0 and actions[1]==0:
        return "-"
    return "H" if actions[0] > actions[1] else "S"

def print_q_table(q_table, filename='q_table_visualization.png', show=False):
    #table size = (704, 2)
    #CHATGPT
    #will store Actions as strings Hit/Stand.
    q_grid_soft = np.empty((32, 11), dtype=object)
    q_grid_hard = np.empty((32, 11), dtype=object)

    for p in range(32):
        for d in range(11):
            idx = p * 11 * 2 + d * 2
            q_grid_soft[p][d] = _get_Action_String(q_table[idx + 1])
            q_grid_hard[p][d] = _get_Action_String(q_table[idx])

    # Focus only on player totals from 8 to 21
    start_row = 8
    end_row = 22  # Python slice is exclusive on the end
    q_grid_soft = q_grid_soft[start_row:end_row, :]
    q_grid_hard = q_grid_hard[start_row:end_row, :]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    color_map = {
        "H": "lightgreen",
        "S": "lightcoral",
        "-": "lightgrey"
    }

    # Plot soft hand
    axes[0].imshow(np.zeros_like(q_grid_soft, dtype=float), cmap='Greys', vmin=0, vmax=1)
    for i in range(end_row - start_row):
        for j in range(11):
            action = q_grid_soft[i][j]
            axes[0].text(
                j, i, action, ha='center', va='center',
                bbox=dict(facecolor=color_map[action], edgecolor='black', boxstyle='round,pad=0.3')
            )
    axes[0].set_title('Actions if hand is soft. A=11 (H: Hit, S: Stand)')
    axes[0].set_xlabel('Dealer Card (1–11)')
    axes[0].set_ylabel('Player Sum (8–21)')
    axes[0].set_xticks(np.arange(11))
    axes[0].set_xticklabels([str(i + 1) for i in range(11)])
    axes[0].set_yticks(np.arange(end_row - start_row))
    axes[0].set_yticklabels([str(i) for i in range(start_row, end_row)])
    axes[0].invert_yaxis()

    # Plot hard hand
    axes[1].imshow(np.zeros_like(q_grid_hard, dtype=float), cmap='Greys', vmin=0, vmax=1)
    for i in range(end_row - start_row):
        for j in range(11):
            action = q_grid_hard[i][j]
            axes[1].text(
                j, i, action, ha='center', va='center',
                bbox=dict(facecolor=color_map[action], edgecolor='black', boxstyle='round,pad=0.3')
            )
    axes[1].set_title('Actions if hand is hard. A=1 (H: Hit, S: Stand)')
    axes[1].set_xlabel('Dealer Card (1–11)')
    axes[1].set_ylabel('Player Sum (8–21)')
    axes[1].set_xticks(np.arange(11))
    axes[1].set_xticklabels([str(i + 1) for i in range(11)])
    axes[1].set_yticks(np.arange(end_row - start_row))
    axes[1].set_yticklabels([str(i) for i in range(start_row, end_row)])
    axes[1].invert_yaxis()

    if show:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close(fig)

    return


    # Save with tab-separated values
    np.savetxt("output.txt", q_table, fmt="%.2f", delimiter="\t", header="Col1\tCol2", comments='')


def moving_average(data, window_size):
    #CHATGPT
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_error(training_error):
    #CHATGPT
    smoothed_error = moving_average(training_error, window_size=100000)

    plt.plot(smoothed_error)
    plt.title('Smoothed Training Error (Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed TD Error')
    plt.grid(True)
    plt.savefig("error.png")
    plt.show()


