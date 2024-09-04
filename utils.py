import matplotlib.pyplot as plt

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
        plt.plot(x_values, y_values[i], label=names[i], color=colors[i]) 
    

    # labels   
    plt.xlabel('Episodes')
    plt.ylabel('Fitness value')
    
    # legend
    plt.legend(fontsize=14)

    # display the plot
    plt.grid(True)
    plt.show()





if __name__=='__main__': 
    #execute_mean()

    episodes=1500
    names=['simple_dqn','dqn']
    input_files=['avg_scores_{}_means.txt'.format(x) for x in names]    
    colors=['red','blue']

    print_2Dplot(episodes, input_files, names, colors)