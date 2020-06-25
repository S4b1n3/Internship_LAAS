# Internship_LAAS

TODO : add compilation instructions (and pre-requirements)
TDOD : add tclap source code here
TODO : update the parameters part since we use tclap
TODO : add parameter for architecture
TODO: add an example of execution:  bin/bnn-main  --nb_examples 1 --seed 323  --index_model 1  1 1 1  
TODO : add parameter to indiquate if we want to check the solution 

bnn-main.cc : is the file that runs the cp model. It takes as parameter:

 */ param1 : boolean which define the type of constraint used  to compute the value of the preactivation

 */ param2 : number of input examples

 */ param3 : index of the model to run

 */ param4 : path of the output files

 */ param4 : number of neurons on the first hidden layer

 */ param5 : number of neurons on the second hidden layer

 */ paramX : number of neurons on the Xth hidden layer

cp_minweight_model.h : is the CP model for the full classification with min weight

data.h : is the data instance class

solution.h : is the solution class

To run the cp model, first compile the bnn-main.cc file from ORTools folder with
  `make build SOURCE=.../bnn-main.cc`

Then use the command from the same folder

  `./bin/bnn-main  [-O <string>] [-C] -X <int> -M <int> [--] [--version][-h] <int> ...`

 where the argument after the flag -X (respectively -M, -O) is param2 (respectively param3, 4) and is mandatory.

If the flag -C is written, the product constraint will be used to compute the preactivation values.

The order of the flags is not important, just be careful to write the value corresponding to a flag just after the right one.

The next int variables are optional and correspond to the other variables. The order that the arguments are added to the command line is the order that they will be parsed and added in the architecture.

Use `./bin/bnn-main --help` for complete usage explanations.

## Result files tree management

All the result files are stored in a folder "results". In this folder, there subfolders named "resultsXN" where X is the sum of the neurons of the hidden layers.

The third level of the tree is the architecture tested by the model : each subsubfolder is named "results\_X1\_..\_Xn" where Xi is the number of neuron on the hidden layer i.

The last level of the tree is the model tested : the subsubsubfolders are intitled "resultsMY" where Y is the index of the model.

The results files are contained by theses subsubsubfolders. The names of the files are defined : "resultsK.stat" where K is the number of input examples.

It is not mandatory to create the file tree before each run, the main will create it if it does not exist and won't raise an error if it exist. The parameter 4 allows you to choose where to put this file tree. 

The parser will then run through the tree. The parser takes as parameter the subfolder corresponding to the number of neurons.
