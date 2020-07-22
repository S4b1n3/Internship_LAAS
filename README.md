# Internship_LAAS

bnn-main.cc : is the file that runs the cp model.

cp_model.h : is the base class for all the models

cp_minweight_model.h : is the CP model for the full classification with min weight

cp_maxclassification_model.h : is the CP model for max classification

cp_minweight_model : is the CP model for max classification with soft version of robustness constraints

data.h : is the data instance class

solution.h : is the solution class

evaluation.h : is the class that tests the solution on the whole testing and training sets

Run intructions :

USAGE:

   ./bin/bnn-main  [-O <string>] [-D <string>] [-F] [-V] [-C] [-A <int>]
                   ...  [-T <double>] [-E <int>] [-K <int>] [-X <int>] [-S
                   <int>] -M <char> [--] [--version] [-h]


Where:

   -O <string>,  --output_file <string>
     Path of the output file

   -D <string>,  --strategy <string>
     The search strategy

   -F,  --evaluation
     indicates if the evaluation on the testing and training sets has to be
     done

   -V,  --check
     indicates if the solution returned has to be tested

   -C,  --product_constraints
     indicates the use of product constraints

   -A <int>,  --archi <int>  (accepted multiple times)
     Architecture of the model

   -T <double>,  --time <double>
     Time limit for the solver

   -E <int>,  --nb_examples_per_label <int>
     Number of examples per label

   -K <int>,  --k <int>
     Robustness parameter

   -X <int>,  --nb_examples <int>
     Number of examples

   -S <int>,  --seed <int>
     Seed

   -M <char>,  --index_model <char>
     (required)  Index of the model to run

   --,  --ignore_rest
     Ignores the rest of the labeled arguments following this flag.

   --version
     Displays version information and exits.

   -h,  --help
     Displays usage information and exits.


If the flag -C is written, the product constraint will be used to compute the preactivation values.

The flag -E builds a dataset composed of the user choice number of random examples from each class. The flag -X builds a dataset composed of the user choice number of random examples. If both (or no one) are used to run the model, a defaultmode is run with one example (equivalent to -X 1).

The order of the flags is not important, just be careful to write the value corresponding to a flag just after the right one.

To complete the architecture of the network, use the flag -A (or --archi) for each hidden layer. The order that the arguments are added to the command line is the order that they will be parsed and added in the architecture.

Some examples of execution :
  ./bin/bnn-main  --index_model 1 --nb_examples 1 --seed 323 --archi 1 --archi 1 --archi 1
  ./bin/bnn-main -M 2 -C -X 3 -A 16 -A 16 -V -F
  ./bin/bnn-main -M 1 -C -E 10 -D antilex_max_0



## Result files tree management

All the result files are stored in a folder "results". In this folder, there subfolders named "resultsMX" (or "resultsMX-C") where X is the index of the model. The flag -C is add if it had been used while the execution of the program.

The third level of the tree is the architecture tested by the model : each subsubfolder is named "results\_X1\_..\_Xn" where Xi is the number of neuron on the hidden layer i.

The last level of the tree is the strategy tested : the subsubsubfolders are intitled "resultslex" where lex is the name of the strategy.

The results files are contained by theses subsubsubfolders. The names of the files are defined : "resultsK.stat" where K is the number of input examples.

It is not mandatory to create the file tree before each run, the main will create it if it does not exist and won't raise an error if it exist. The parameter 4 allows the user to choose where to put this file tree.

The parser will then run through the tree. The parser takes as parameter the subfolder corresponding to the architecture.
