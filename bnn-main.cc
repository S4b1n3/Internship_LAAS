
#include "data.h"
#include "solution.h"
#include "cp_minweight_model.h"
#include "cp_maxclassification_model.h"
#include "evaluation.h"

#include "/home/sabine/Documents/Seafile/Stage LAAS/or-tools_Ubuntu-18.04-64bit_v7.5.7466/tclap/CmdLine.h"

#include <string>
#include <vector>

using namespace TCLAP;

int _index_model;
int _nb_examples;
std::vector<int> architecture;
int _nb_neurons;
bool _prod_constraint;
std::string _output_path;

void parseOptions(int argc, char** argv);

int main(int argc, char **argv) {

  srand(time(NULL));

  architecture.push_back(784);
  parseOptions(argc, argv);
  architecture.push_back(10);

  std::string filename;


  filename.append(_output_path);

  filename.append("/results/results"+std::to_string(_nb_neurons)+"N/results");


  for (size_t i = 1; i < architecture.size()-1; i++) {
    filename.append("_"+std::to_string(architecture[i]));
  }
  if (architecture.size()-2 == 0) {
    filename.append("_0");
  }

  filename.append("/resultsM"+std::to_string(_index_model));

  std::string cmd("mkdir -p "+filename);
  system(cmd.c_str());

  std::cout << filename << std::endl;

  double accuracy_train, accuracy_test;
  switch (_index_model) {
    case 1:
      {
        operations_research::sat::CPModel_MinWeight first_model(architecture, _nb_examples, _prod_constraint, filename);
        std::cout<<std::endl<<std::endl;
        first_model.run(1200.0) ;
        first_model.print_statistics();
        //first_model.print_solution(first_model.get_response());
        //first_model.print_solution_bis(first_model.get_response());
        //first_model.print_all_solutions() ;
        Evaluation test(100, first_model.get_weights_solution(), architecture);
        accuracy_test = test.run_evaluation_test_set();
        accuracy_train = test.run_evaluation_train_set();
        break;
      }
    case 2:
      {
        operations_research::sat::CPModel_MaxClassification second_model(architecture, _nb_examples, _prod_constraint, filename);
        std::cout<<std::endl<<std::endl;
        second_model.run(1200.0);
        second_model.print_statistics();
        Evaluation test(100, second_model.get_weights_solution(), architecture);
        accuracy_test = test.run_evaluation_test_set();
        accuracy_train = test.run_evaluation_train_set();
        break;
      }
    default:
      {
        std::cout << "There is no model with index "<< _index_model << '\n';
        std::cout << "Please select 1 or 2" << '\n';
      }

  }

  std::cout << "Testing accuracy of the model : "<< accuracy_test << '\n';
  std::cout << "Training accuracy of the model : "<< accuracy_train << '\n';
  std::string result_file = filename+"/results"+std::to_string(nb_examples)+".stat";
  std::ofstream results(result_file.c_str(), std::ios::app);
  results << "test accuracy " << accuracy_test << std::endl;
  results << "train accuracy " << accuracy_train << std::endl;

  return EXIT_SUCCESS;
}

void parseOptions(int argc, char** argv)
{
	try {

	CmdLine cmd("BNN Parameters", ' ', "0.99" );

	//
	// Define arguments
	//
  ValueArg<int> imodel ("M", "index_model", "Index of the model to run", true, 1, "int");
  cmd.add(imodel);

	ValueArg<int> nb_ex("X", "nb_examples", "Number of examples", true, 1, "int");
	cmd.add(nb_ex);

	UnlabeledMultiArg<int> archi("archi", "Architecture of the model", false, "int");
	cmd.add(archi);

  SwitchArg bool_prod("C","use_prod_constraint", "bool tests the use of product constraints", false);
	cmd.add(bool_prod);

  ValueArg<std::string> out_file("O", "output_file", "Path of the output file", false, "BNN", "string");
  cmd.add(out_file);

	//
	// Parse the command line.
	//
	cmd.parse(argc,argv);

	//
	// Set variables
	//
  _index_model = imodel.getValue();
	_nb_examples = nb_ex.getValue();
  _prod_constraint = bool_prod.getValue();
  _output_path = out_file.getValue();

	std::vector<int> v = archi.getValue();
	for ( int i = 0; static_cast<unsigned int>(i) < v.size(); i++ ){
      architecture.push_back(v[i]);
      _nb_neurons += v[i];
  }

	} catch ( ArgException& e )
	{ std::cout << "ERROR: " << e.error() << " " << e.argId() << std::endl; }
}
