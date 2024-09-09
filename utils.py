import os
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------------------------------------------------------------------------ 
# -- PLOTS --------------------------------------------------------------------------- 

"""
Calculate the mean of a given file.txt score results.

Args:
    input_file (string)  : Name of the file which is going to be readed.
    output_file (string) : Name of the file where the means are going to be stored.
    mean_number (int)    : Numbers of values to calculate the mean.
"""
def calculate_mean(input_file, output_file, mean_number):    
    values=[0 for _ in range(mean_number)]
    sum=0
    size=0
    
    means=[]
    # dqn: segundos:        2763.1512389
    # simple_dqn: segundos: 2748.699877023697

    with open(input_file, 'r') as file:
        for line in file:
            # Convert each line to a float
            number=float(line.strip())
            
            if size>=100: sum-=values[size%100]

            values[size%100]=number
            size+=1
            sum+=number

            if size>=mean_number: means.append(sum/mean_number)
            else: means.append(sum/size)
            
   

    
    with open(output_file, 'w') as file:
        for val in means:
            file.write(f"{val:.6f}\n")

def execute_mean():
    input_file='avg_scores_dqn.txt' 
    output_file='avg_scores_dqn_means.txt'  
    calculate_mean(input_file, output_file,100)

    input_file='avg_scores_simple_dqn.txt'  
    output_file='avg_scores_simple_dqn_means.txt' 
    calculate_mean(input_file, output_file,100)

"""
Read all the values in a file .txt. 

Format:
    One float per row.

Args:   
    filename (string) : Name of the file which is going to be readed.
"""
def read_file_values_1_row(filename):
    with open(filename, 'r') as file:
        # Read all lines, convert them to floats, and return as a list
        return [float(line.strip()) for line in file]

"""
Prints the 2D-Plot generated with the input files values.

Args:
    episodes (int)          : Number of episodes.
    input_files (string[])  : Names of the files which contains the values of the functions.
    names (string[])        : Names of the algorithms.
    colors (string[])       : Colors of the functions in the 2D-Plot.
    
"""
def print_2Dplot(episodes, input_files, names, colors):
    
    # reading values
    y_values=[read_file_values_1_row(name) for name in input_files]      

    # Generate x-values from 1 to 1500
    x_values = list(range(1, episodes+1))

    # Plot the functions
    # size 
    plt.figure(figsize=(10, 6))

    

    # add functions
    for i in range(len(names)):
        plt.plot(x_values, y_values[i][:episodes], label=names[i], color=colors[i]) 
    

    # labels   
    plt.xlabel('Episodes')
    plt.ylabel('Fitness value')
    
    # legend
    plt.legend(fontsize=14)

    # display the plot
    plt.grid(True)
    plt.show()

# ------------------------------------------------------------------------------------ 
# -- DQN ----------------------------------------------------------------------------- 

"""
Store the results of the search function.

Args:
    fc_dim (int)        : Size of the Fully Connected layers.
    eps_dec (float)     : Decreasing number per iteration of epsilon.
    lr (float)          : Learning rate.   
    avg_score (float)   : Average score of the last 100 episodes.
    done (boolean)      : Finalization variable.
    time (float)        : Execution time.
    eps (int)           : Id number of the episode.   
"""
def store_result(fc_dim, eps_dec, lr, avg_score, done, time, eps):
    message = 'fc_dim: {}, eps_dec: {}, lr: {:.5f}, avg_score: {:.2f}, done: {}, time: {:.2f}, episodes: {}'.format(
        fc_dim, eps_dec, lr, avg_score, done, time, eps
    )

    try:
        with open("result2.txt", 'a') as file:
            file.write(message + '\n')
    except Exception as e:
        print(f"An error occurred: {e}")


"""
Result the average scores of the training session.

Args:
    avg_score (float)   : Average score of the last 100 episodes.
    algorithm (string)  : Name of the used algorithm.
"""
def store_avg_score(avg_score, algorithm):
    try:
        with open("avg_scores_{}.txt".format(algorithm), 'a') as file:
            file.write(str(avg_score) + '\n')
    except Exception as e:
        print(f"An error occurred: {e}")
  
# ------------------------------------------------------------------------------------ 
# -- PPO ----------------------------------------------------------------------------- 

"""
Store the best score obtained in a learning session.

Args:
    best_score_ppo (string) : Path to the .txt file.
    score (float)           : Obtained score.
"""
def store_best_score(best_score_path, score):
    try:
        with open(best_score_path, 'w') as file:
            file.write(str(score))
    except Exception as e:
        print(f"An error occurred: {e}")

"""
Load the best score stored in a .txt file, 
    from a previous learning session.

Args:
    best_score_ppo (string) : Path to the .txt file.

Return:
    ret (float) : Stored best score.
"""
def load_best_score(best_score_path):
    ret=0

    try:
        with open(best_score_path, 'r') as file:
            ret=float(file.readline())
    except Exception as e:
        print(f"An error occurred: {e}")
    
    print("Best Score: {}\n".format(ret))
    return ret


# ------------------------------------------------------------------------------------ 
# -- FILES --------------------------------------------------------------------------- 

def extract_avg_score(input_filename, output_filename, avg_name):
    avg_scores=[]
    idx=-1
    
    try:
        with open(input_filename, 'r') as infile:
            count=0
            for line in infile:                
                parts=line.split()
                for part in parts:
                    if part==avg_name:
                        break
                    count+=1
                break
            idx=count+1
            
            
            for line in infile:
                aux=0

                parts=line.split()
                if parts[0]=='Time': aux=2
                if parts[3]=='SAVING': continue
                
                # Ensure there are enough parts in the line
                if len(parts)>=idx:   
                    """print(parts) """                
                    avg_scores.append(float(parts[idx+aux]))
                                            
        
        with open(output_filename, 'w') as outfile:
            for x in avg_scores:
                outfile.write(f"{x}\n")

    except FileNotFoundError:
        print(f"File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def flatten_matrix(matrix):
    return [x for row in matrix for x in row]


def read_last_val(output_file='output.txt'):
    last_floats = []
    
    # Get all .txt files in the current directory
    txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
    
    # Iterate over all the .txt files
    for file_name in txt_files:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            if lines:  # Check if the file is not empty
                # Split the last line into words and try to get the last float
                try:
                    last_line = lines[-1].strip().split()
                    last_float = float(last_line[-1])
                    last_floats.append(last_float)
                except (ValueError, IndexError):
                    print(f"Could not extract float from file: {file_name}")
    
    # Store the extracted floats in a new file
    with open(output_file, 'w') as output:
        for value in last_floats:
            output.write(f"{value}\n")
    
    print(f"Stored last floats from {len(txt_files)} files in {output_file}.")


def plot3D():

    root = tk.Tk()
    root.title("3x3 Histograms GUI")
    
    # Read data from the file (72 float values)
    file_name = "output.txt"  # Change this to your file's path
    
    with open(file_name, 'r') as file:
        data = [float(line.strip()) for line in file.readlines()]

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    
    # Calculate Y axis bounds (min, max) for all histograms
    y_min = np.min(data)-10
    y_max = np.max(data)+10
    

    lr_vals=['1e-3', '2.5e-3', '5e-3', 
             '1e-4', '2.5e-4', '5e-4', 
             '1e-5', '2.5e-5', '5e-5']
    colors = ['#ff0000', '#ec008c', '#0000ff', 
              '#ff8000', '#000000', '#00adef', 
              '#248714', '#33FF96']
    
    # Create histograms for each subplot
    for i, ax in enumerate(axs.flat):
        start_idx = i * 8
        end_idx = start_idx + 8
        # Adjust bar positions by shrinking the spacing between them
        bar_positions = np.arange(8)  # Position each bar closer together
        ax.bar(bar_positions, data[start_idx:end_idx], color=colors, width=0.5)  # Slightly wider bars
        ax.set_ylim([y_min, y_max])  # Set the same Y axis bounds
        ax.set_xticks(bar_positions)  # Keep the bar positions aligned
        ax.set_title(f'LR = {lr_vals[i]}')
    
    # Display the figure in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


    # Run the Tkinter main loop
    root.mainloop()

# ------------------------------------------------------------------------------------ 
# -- MAIN ---------------------------------------------------------------------------- 

if __name__=='__main__': 
    """execute_mean()"""

    """for i in range(9):
        for j in range(8):
            name='scores_{}_{}'.format(i,j)                
            calculate_mean(name+'.txt',name+'_mean.txt',100)"""
    
    """read_last_val()"""
    plot3D()

    """episodes=1500
    names=['simple_dqn','dqn', 'ppo']
    input_files=['avg_scores_{}_means.txt'.format(x) for x in names]    
    colors=['red','blue', 'green']

    print_2Dplot(episodes, input_files, names, colors)"""

