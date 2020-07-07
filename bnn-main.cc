
#include "data.h"
#include "solution.h"
#include "cp_minweight_model.h"
#include "cp_maxclassification_model.h"
#include "cp_maxclassification_model2.h"
#include "cp_maxsum_weights_example_model.h"
#include "cp_robust_model.h"
#include "evaluation.h"

#include "tclap/CmdLine.h"

#include <string>
#include <vector>

using namespace TCLAP;

int _index_model;
int _seed;
int _nb_examples;
int _nb_examples_per_label;
int _k;
double _time;
std::vector<int> architecture;
int _nb_neurons;
bool _prod_constraint;
std::string _strategy;
std::string _output_path;
bool _check_solution;

void parseOptions(int argc, char** argv);

int main(int argc, char **argv) {

	architecture.push_back(784);
	parseOptions(argc, argv);
	architecture.push_back(10);

	srand(_seed);
	std::string filename;

	filename.append(_output_path);

	if(_prod_constraint)
		filename.append("/results/resultsM"+std::to_string(_index_model)+"-C/results");
	else
		filename.append("/results/resultsM"+std::to_string(_index_model)+"/results");

	for (size_t i = 1; i < architecture.size()-1; i++) {
		filename.append("_"+std::to_string(architecture[i]));
	}
	if (architecture.size()-2 == 0) {
		filename.append("_0");
	}

	filename.append("/");

	std::string cmd("mkdir -p "+filename);
	system(cmd.c_str());

	std::cout << filename << std::endl;

	double accuracy_train, accuracy_test;

	switch (_index_model) {
	case 1:
	{
		if (_nb_examples == 0 && _nb_examples_per_label != 0){
			operations_research::sat::CPModel_MinWeight model(_nb_examples_per_label, architecture, _prod_constraint, filename);
			model.run(_time, _strategy);
			int status = model.print_statistics(_check_solution, _strategy);
			if(status == 2 || status == 4){
				Evaluation test(model.get_weights_solution(), model.get_data());
				accuracy_test = test.run_evaluation(true);
				accuracy_train = test.run_evaluation(false);
			}
		 }
		else{
			if (_nb_examples != 0 && _nb_examples_per_label == 0) {
				operations_research::sat::CPModel_MinWeight model(architecture, _nb_examples, _prod_constraint, filename);
				model.run(_time, _strategy);
				int status = model.print_statistics(_check_solution, _strategy);
				if(status == 2 || status == 4){
					Evaluation test(model.get_weights_solution(), model.get_data());
					accuracy_test = test.run_evaluation(true);
					accuracy_train = test.run_evaluation(false);
				}
			}
			else{
				std::cout << "Invalid number of examples : default mode launched" << '\n';
				operations_research::sat::CPModel_MinWeight model(architecture, 1, _prod_constraint, filename);
				model.run(_time, _strategy);
				int status = model.print_statistics(_check_solution, _strategy);
				if(status == 2 || status == 4){
					Evaluation test(model.get_weights_solution(), model.get_data());
					accuracy_test = test.run_evaluation(true);
					accuracy_train = test.run_evaluation(false);
				}
			}
		}
		//first_model.print_solution(first_model.get_response());
		//first_model.print_solution_bis(first_model.get_response());
		//first_model.print_all_solutions() ;
		break;
	}
	case 2:
	{
		if (_nb_examples == 0 && _nb_examples_per_label != 0){
			operations_research::sat::CPModel_MaxClassification model(_nb_examples_per_label, architecture, _prod_constraint, filename);
			model.run(_time, _strategy);
			int status = model.print_statistics(_check_solution, _strategy);
			if(status == 2 || status == 4){
				Evaluation test(model.get_weights_solution(), model.get_data());
				accuracy_test = test.run_evaluation(true);
				accuracy_train = test.run_evaluation(false);
			}
		}

		else{
			if (_nb_examples != 0 && _nb_examples_per_label == 0) {
				operations_research::sat::CPModel_MaxClassification model(architecture, _nb_examples, _prod_constraint, filename);
				model.run(_time, _strategy);
				int status = model.print_statistics(_check_solution, _strategy);
				if(status == 2 || status == 4){
					Evaluation test(model.get_weights_solution(), model.get_data());
					accuracy_test = test.run_evaluation(true);
					accuracy_train = test.run_evaluation(false);
				}
			}
			else{
				std::cout << "Invalid number of examples : default mode launched" << '\n';
				operations_research::sat::CPModel_MaxClassification model(architecture, 1, _prod_constraint, filename);
				model.run(_time, _strategy);
				int status = model.print_statistics(_check_solution, _strategy);
				if(status == 2 || status == 4){
					Evaluation test(model.get_weights_solution(), model.get_data());
					accuracy_test = test.run_evaluation(true);
					accuracy_train = test.run_evaluation(false);
				}
			}
		}
		break;
	}
	case 3:
	{
		if (_nb_examples == 0 && _nb_examples_per_label != 0){
			operations_research::sat::CPModel_MaxClassification2 model(_nb_examples_per_label, architecture, _prod_constraint, filename);
			model.run(_time, _strategy);
			int status = model.print_statistics(_check_solution, _strategy);
			if(status == 2 || status == 4){
				Evaluation test(model.get_weights_solution(), model.get_data());
				accuracy_test = test.run_evaluation(true);
				accuracy_train = test.run_evaluation(false);
			}
		}

		else{
			if (_nb_examples != 0 && _nb_examples_per_label == 0) {
				operations_research::sat::CPModel_MaxClassification2 model(architecture, _nb_examples, _prod_constraint, filename);
				model.run(_time, _strategy);
				int status = model.print_statistics(_check_solution, _strategy);
				if(status == 2 || status == 4){
					Evaluation test(model.get_weights_solution(), model.get_data());
					accuracy_test = test.run_evaluation(true);
					accuracy_train = test.run_evaluation(false);
				}
			}
			else{
				std::cout << "Invalid number of examples : default mode launched" << '\n';
				operations_research::sat::CPModel_MaxClassification2 model(architecture, 1, _prod_constraint, filename);
				model.run(_time, _strategy);
				int status = model.print_statistics(_check_solution, _strategy);
				if(status == 2 || status == 4){
					Evaluation test(model.get_weights_solution(), model.get_data());
					accuracy_test = test.run_evaluation(true);
					accuracy_train = test.run_evaluation(false);
				}
			}
		}
		break;
	}
	case 4:
	{
		operations_research::sat::CPModel_MaxSum third_model(architecture, _nb_examples, _prod_constraint, filename);
		std::cout<<std::endl<<std::endl;
		third_model.run(_time ,  _strategy) ;
		int status = third_model.print_statistics(_check_solution, _strategy);
		if(status == 2 || status == 4){
			Evaluation test(third_model.get_weights_solution(), third_model.get_data());
			accuracy_test = test.run_evaluation(true);
			accuracy_train = test.run_evaluation(false);
		}
		break;
	}
	case 5:
	{
		operations_research::sat::CPModel_Robust model(architecture, _nb_examples, _prod_constraint, filename, _k);
		std::cout<<std::endl<<std::endl;
		model.run(_time ,  _strategy) ;
		int status = model.print_statistics(_check_solution, _strategy);
		if(status == 2 || status == 4){
			Evaluation test(model.get_weights_solution(), model.get_data());
			accuracy_test = test.run_evaluation(true);
			accuracy_train = test.run_evaluation(false);
		}
		break;
	}
	default:
	{
		std::cout << "There is no model with index "<< _index_model << '\n';
		std::cout << "Please select 1, 2, 3 or 4, 5" << '\n';
	}

	}
	if (accuracy_test < 0.1) {
		accuracy_test = 0;
	}
	if (accuracy_train < 0.1) {
		accuracy_train = 0;
	}

	std::cout << "Testing accuracy of the model : "<< std::round(accuracy_test) << '\n';
	std::cout << "Training accuracy of the model : "<< std::round(accuracy_train) << '\n';
	std::string result_file = filename+"/results_"+_strategy+".stat";
	std::ofstream results(result_file.c_str(), std::ios::app);
	results << "test accuracy " << accuracy_test << std::endl;
	results << "train accuracy " << accuracy_train << std::endl;

	return EXIT_SUCCESS;
}

void parseOptions(int argc, char** argv){
	try {

		CmdLine cmd("BNN Parameters", ' ', "0.99" );


		//
		// Define arguments
		//
		// TODO: change this argument to be a string instead
		ValueArg<int> imodel ("M", "index_model", "Index of the model to run", true, 1, "int");
		cmd.add(imodel);

		ValueArg<int> seed ("S", "seed", "Seed", false, 1, "int");
		cmd.add(seed);

		ValueArg<int> nb_ex("X", "nb_examples", "Number of examples", false, 0, "int");
		cmd.add(nb_ex);

		ValueArg<int> param_k("K", "k", "Robustness parameter", false, 1, "int");
		cmd.add(param_k);

		ValueArg<int> nb_ex_per_label("E", "nb_examples_per_label", "Number of examples per label", false, 0, "int");
		cmd.add(nb_ex_per_label);

		ValueArg<double> time("T", "time", "Time limit for the solver", false, 1200.0, "double");
		cmd.add(time);

		MultiArg<int> archi("A", "archi", "Architecture of the model", false, "int");
		cmd.add(archi);

		SwitchArg bool_prod("C","product_constraints", "indicates the use of product constraints", false);
		cmd.add(bool_prod);

		SwitchArg check_sol("V","check", "indicates if the solution returned has to be tested", false);
		cmd.add(check_sol);

		ValueArg<std::string> search_strategy("D", "strategy", "The search strategy", false, "lex", "string");
		cmd.add(search_strategy);

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
		_seed = seed.getValue();
		_nb_examples = nb_ex.getValue();
		_nb_examples_per_label = nb_ex_per_label.getValue();
		_k = param_k.getValue();
		_time = time.getValue();
		_prod_constraint = bool_prod.getValue();
		_strategy =search_strategy.getValue();
		_output_path = out_file.getValue();
		_check_solution = check_sol.getValue();

		std::vector<int> v = archi.getValue();
		for ( int i = 0; static_cast<unsigned int>(i) < v.size(); i++ ){
			architecture.push_back(v[i]);
			_nb_neurons += v[i];
		}
	} catch ( ArgException& e )
	{ std::cout << "ERROR: " << e.error() << " " << e.argId() << std::endl; }
}
