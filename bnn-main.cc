
#include "data.h"
#include "solution.h"
#include "cp_minweight_model.h"
#include "cp_maxclassification_model.h"

#include "/home/sabine/Documents/Seafile/Stage LAAS/or-tools_Ubuntu-18.04-64bit_v7.5.7466/tclap/CmdLine.h"

#include <string>
#include <vector>

using namespace TCLAP;

int _nb_examples;
std::vector<int> architecture;
int _nb_neurons;
bool _prod_constraint;

void parseOptions(int argc, char** argv);

int main(int argc, char **argv) {

  srand(time(NULL));

  architecture.push_back(784);
  parseOptions(argc, argv);
  architecture.push_back(10);

  std::string filename("BNN/results/results"+std::to_string(_nb_neurons)+"N/results");
  for (size_t i = 1; i < architecture.size()-1; i++) {
    filename.append("_"+std::to_string(architecture[i]));
  }

  filename.append("/results"+std::to_string(_nb_examples)+".stat");

  std::cout << filename << std::endl;

  operations_research::sat::CPModel_MinWeight first_model(architecture, _nb_examples, _prod_constraint);
  //operations_research::sat::CPModel_MaxClassification second_model(archi_test, nb_examples);

  std::cout<<std::endl<<std::endl;

  //second_model.run(1200.0) ;
  first_model.run(1200.0) ;
  first_model.print_statistics(filename);

  //second_model.print_statistics(filename) ;
  //second_model.print_solution(second_model.get_response());

  //first_model.print_solution_bis(first_model.get_response());
  //first_model.print_all_solutions() ;
  return EXIT_SUCCESS;
}

void parseOptions(int argc, char** argv)
{
	try {

	CmdLine cmd("BNN Parameters", ' ', "0.99" );
  
	//
	// Define arguments
	//

	ValueArg<int> itest("X", "nb_examples", "Number of examples", true, 1, "int");
	cmd.add( itest );

	UnlabeledMultiArg<int> mtest("archi", "Architecture of the model", false, "int");
	cmd.add( mtest );

  SwitchArg btest("C","use_prod_constraint", "bool tests the use of product constraints", false);
	cmd.add( btest );

	//
	// Parse the command line.
	//
	cmd.parse(argc,argv);

	//
	// Set variables
	//
	_nb_examples = itest.getValue();
  _prod_constraint = btest.getValue();

	std::vector<int> v = mtest.getValue();
	for ( int i = 0; static_cast<unsigned int>(i) < v.size(); i++ ){
      architecture.push_back(v[i]);
      _nb_neurons += v[i];
  }

	} catch ( ArgException& e )
	{ std::cout << "ERROR: " << e.error() << " " << e.argId() << std::endl; }
}
