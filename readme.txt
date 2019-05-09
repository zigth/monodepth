The code for this project was made with python, using Tensorflow, version 1.6.0 and Keras, version 2.1.6. 

The python file textNN.py is used to build a neural network, stored as thought_model.h5, for deciding which thoughts are useful. 
This requires a certain amount of data to work with, preferably a large amount, which should be stored as follows:

1 textfile containing a list of useful thoughts for reference (1 per line in the file)
1 textfile containing a list of useless thoughts for comparision (again 1 per line)
1 textfile containing some useful thoughts, not contained in the other file, for validation (again 1 per line)
1 textfile containing some useless thoughts, not contained in the other file, for validation (again 1 per line)

these files should be stored under these directories and given the right names:
data/training/useful.txt
data/training/useless.txt
data/validation/useful.txt
data/validation/useless.txt

Alternatively the directory names in textNN.py and thoughtGenerator.py need to be changed. 

The python file thoughtGenerator.py then loads the neural network, asks the user for a sentence and prints out the permutations that the network recognizes as useful.

This approach has the advantage that the nature of what qualifies a thought as useful is solely determined through the selection of examples by the user. 

