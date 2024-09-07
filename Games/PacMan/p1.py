import sys

with open("prueba.txt", 'r') as file:        
    for line in file:
        row=line.split()
        if row[0]==row[2] and row[1]==row[3]: 
            print("IGUAL")
        