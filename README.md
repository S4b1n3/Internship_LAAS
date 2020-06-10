# Internship_LAAS

bnn-main.cc : is the file that runs the cp model. It takes as parameter: 

 */ param1 : number of input examples 
 
 */ param2 : number of neurons on the first hidden layer
 
 */ param3 : number of neurons on the second hidden layer
 
 */ paramX : number of neurons on the Xth hidden layer
 
cp_minweight_model.h : is the CP model for the full classification with min weight 

data.h : is the data instance class

solution.h : is the solution class 

To run the cp model, first compile the bnn-main.cc file from ORTools folder with 
  `make build SOURCE=.../bnn-main.cc`
  
Then use the command from the same folder
  `./bin/bnn_main.cc -X <int> [--] [--version] [-h] <int> ... `
 where the first argument is param1 and is mandatory.
The next int variables are optional and correspond to the other variables. The order that the arguments are added to the command line is the order that they will be parsed and added in the architecture.
Use `./bin/bnn-main --help` for complete usage explanations.

## Result files tree management

All the result files are stored in a folder "results". In this folder, there subfolders named "resultsXN" where X is the sum of the neurons of the hidden layers.
The third level of the tree is the architecture tested by the model : each subsubfolder is named "results\_X1\_..\_Xn" where Xi is the number of neuron on the hidden layer i. 
The results files are contained by theses subsubfolders. The names of the files are defined : "resultsK.stat" where K is the number of input examples.
Before each run, the tree has to be create as above.

The parser will then run through the tree. The parser takes as parameter the subfolder corresponding to the number of neurons.
