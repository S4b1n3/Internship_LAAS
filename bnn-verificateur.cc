
#include "data.h"
#include "solution.h"
#include "cp_minweight_model.h"
#include "cp_maxclassification_model.h"
#include "cp_maxclassification_model2.h"
#include "cp_maxsum_weights_example_model.h"
//#include "cp_robust_model.h"
#include "cp_satisfaction_model.h"
#include "evaluation.h"
#include "get_dataset.h"
#include "tclap/CmdLine.h"

#include <string>
#include <vector>
#include <fstream>


using namespace TCLAP;

std::string _sol_path;
std::string _data_path;
char _index_model;
int _seed;
int _nb_examples;
int _nb_examples_per_label;
int _k;
double _time;
std::vector<int> architecture;
int _nb_neurons;
bool _prod_constraint;
//This is no longer used : --> Remove it's usage
std::string _strategy;
//Use specific search strategy ?
bool _search_strategy ;
//Use layer per layer branching
bool _per_layer_branching ;
std::string _variable_heuristic ;
std::string	_value_heuristic ;
// Use automatic search ?
int _automatic ;

std::string _output_path;
bool _check_solution;
bool _print_solution;
bool _eval;
std::string _input_file;

std::vector<std::vector<std::vector<int>>> weights_temp;

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
	std::string cmd("mkdir -p "+filename);
	system(cmd.c_str());


	filename.append("/results_"+_strategy+".stat");


	double accuracy_train, accuracy_test, accuracy_train_bis, accuracy_test_bis;

	std::vector<std::vector<std::vector<int>>> weights;
	Data bnn_data(architecture);
	int status;

	operations_research::sat::Search_parameters search (_search_strategy , _per_layer_branching , _variable_heuristic , _value_heuristic , _automatic) ;


  switch (_index_model) {
		case '0':
    {
      if (_sol_path != "solution" && _data_path != "dataset") {
        if(!fexists(_data_path)){
            std::cout << " c creating dataset" << '\n';
            correct(_data_path, _sol_path, architecture);
        }
        operations_research::sat::CPModel_Satisfaction model(bnn_data, _prod_constraint, filename, _data_path, _sol_path);
        model.run(_time, search);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
				if (_print_solution)
					model.print_solution(model.get_response(), _print_solution);
      }
      if (_sol_path == "solution" && _data_path == "dataset" && _nb_examples == 0 && _nb_examples_per_label == 0) {
        std::vector<int> correct_examples;
        Evaluation test(weights_temp, bnn_data);
        correct_examples = test.get_correct_examples();
        operations_research::sat::CPModel_Satisfaction model(bnn_data, _prod_constraint, filename, weights_temp, correct_examples);
        model.run(_time, search);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
				if (_print_solution)
					model.print_solution(model.get_response(), _print_solution);
      }
      break;
    }
    case '1':
    {
      if (_sol_path != "solution" && _data_path != "dataset") {
        if(!fexists(_data_path)){
            std::cout << " c creating dataset" << '\n';
            correct(_data_path, _sol_path, architecture);
        }
        operations_research::sat::CPModel_MinWeight model(bnn_data, _prod_constraint, filename, _data_path, _sol_path);
        model.run(_time, search);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
				if (_print_solution)
					model.print_solution(model.get_response(), _print_solution);
      }
      if (_sol_path == "solution" && _data_path == "dataset" && _nb_examples == 0 && _nb_examples_per_label == 0) {
        std::vector<int> correct_examples;
        Evaluation test(weights_temp, bnn_data);
        correct_examples = test.get_correct_examples();
        operations_research::sat::CPModel_MinWeight model(bnn_data, _prod_constraint, filename, weights_temp, correct_examples);
        model.run(_time, search);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
				if (_print_solution)
					model.print_solution(model.get_response(), _print_solution);
      }
      break;
    }
    case '2':
    {
      if (_sol_path != "solution" && _data_path != "dataset") {
        if(!fexists(_data_path)){
            std::cout << " c creating dataset" << '\n';
            correct(_data_path, _sol_path, architecture);
        }
        operations_research::sat::CPModel_MaxClassification model(bnn_data, _prod_constraint, filename, _data_path, _sol_path);
        model.run(_time, search);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
				if (_print_solution)
					model.print_solution(model.get_response(), _print_solution);
      }
      if (_sol_path == "solution" && _data_path == "dataset" && _nb_examples == 0 && _nb_examples_per_label == 0) {
        std::vector<int> correct_examples;
        Evaluation test(weights_temp, bnn_data);
        correct_examples = test.get_correct_examples();
        operations_research::sat::CPModel_MaxClassification model(bnn_data, _prod_constraint, filename, weights_temp, correct_examples);
        model.run(_time, search);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
				if (_print_solution)
					model.print_solution(model.get_response(), _print_solution);
      }
      break;
    }
    case '3':
    {
      if (_sol_path != "solution" && _data_path != "dataset") {
        if(!fexists(_data_path)){
            std::cout << " c creating dataset" << '\n';
            correct(_data_path, _sol_path, architecture);
        }
        operations_research::sat::CPModel_MaxClassification2 model(bnn_data, _prod_constraint, filename, _data_path, _sol_path);
        model.run(_time, search);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
				if (_print_solution)
					model.print_solution(model.get_response(), _print_solution);
      }
      if (_sol_path == "solution" && _data_path == "dataset" && _nb_examples == 0 && _nb_examples_per_label == 0) {
        std::vector<int> correct_examples;
        Evaluation test(weights_temp, bnn_data);
        correct_examples = test.get_correct_examples();
        operations_research::sat::CPModel_MaxClassification2 model(bnn_data, _prod_constraint, filename, weights_temp, correct_examples);
        model.run(_time, search);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
				if (_print_solution)
					model.print_solution(model.get_response(), _print_solution);
      }
      break;
    }
    default:
  	{
  		std::cout << " c There is no model with index "<< _index_model << '\n';
  		std::cout << " c Please select 1, 2, 3 or 4, 5" << '\n';
  	}
  }

	if (_eval) {
		if(status == 2 || status == 4){
			std::cout << " c starting evaluation..." << '\n';
			Evaluation test(weights, bnn_data, filename);
			std::cout << " c Testing accuracy with strong classification criterion : "<< '\n';
			accuracy_test = test.run_evaluation(true, true);
			std::cout << " c Training accuracy with strong classification criterion : "<< '\n';
			accuracy_train = test.run_evaluation(false, true);
			std::cout << " c Testing accuracy with weak classification criterion : "<< '\n';
			accuracy_test_bis = test.run_evaluation(true, false);
			std::cout << " c Training accuracy with weak classification criterion : "<< '\n';
			accuracy_train_bis = test.run_evaluation(false, false);

		if (accuracy_test < 0.1) {
			accuracy_test = 0;
		}
		if (accuracy_train < 0.1) {
			accuracy_train = 0;
		}
		if (accuracy_test_bis < 0.1) {
			accuracy_test_bis = 0;
		}
		if (accuracy_train_bis < 0.1) {
			accuracy_train_bis = 0;
		}

		std::string result_file = filename+"/results_"+_strategy+".stat";
		std::ofstream results(result_file.c_str(), std::ios::app);
		results << "d TEST_STRONG_ACCURACY " << accuracy_test << std::endl;
		results << "d TRAIN_STRONG_ACCURACY " << accuracy_train << std::endl;
		results << "d TEST_WEAK_ACCURACY " << accuracy_test_bis << std::endl;
		results << "d TRAIN_WEAK_ACCURACY " << accuracy_train_bis << std::endl;
		results.close();

		std::cout << " d TEST_STRONG_ACCURACY " << accuracy_test << std::endl;
		std::cout << " d TRAIN_STRONG_ACCURACY " << accuracy_train << std::endl;
		std::cout << " d TEST_WEAK_ACCURACY " << accuracy_test_bis << std::endl;
		std::cout << " d TRAIN_WEAK_ACCURACY " << accuracy_train_bis << std::endl;
	}
	}
		else
			std::cout << " c starting evaluation..." << '\n';

  return EXIT_SUCCESS;
}

void parseOptions(int argc, char** argv){
	try {
		CmdLine cmd("BNN Checker Parameters", ' ', "0.99" );
		//
		// Define arguments
		//

    ValueArg<std::string> isolution ("L", "solution", "Path of the file containing the solution", false, "solution", "string");
		cmd.add(isolution);

    ValueArg<std::string> idata ("Z", "dataset", "Path of the file containing the dataset", false, "dataset", "string");
		cmd.add(idata);

		ValueArg<char> imodel ("M", "index_model", "Index of the model to run", true, '1', "char");
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

		ValueArg<int> print_sol("P","print", "indicates if the solution returned has to be printed, and to which level : 0 nope, 1 minimal, 2 full", false, 0, "int");
		cmd.add(print_sol);

		SwitchArg eval("F","evaluation", "indicates if the evaluation on the testing and training sets has to be done", false);
		cmd.add(eval);

		//Search parameters:
		SwitchArg search_strategy("D", "specific_strategy", "Use a specific search strategy", false);
		cmd.add(search_strategy);

		SwitchArg per_layer_branching("B", "per_layer", "branch per layer", false);
		cmd.add(per_layer_branching);


		ValueArg<std::string> variable_heuristic("H", "var_heuristic", "variable heuristic: lex, min_domain, none", false, "none", "string");
		cmd.add(variable_heuristic);

		ValueArg<std::string> value_heuristic("G", "value_heuristic", "value heuristic: max, min, median, none ", false, "none", "string");
		cmd.add(value_heuristic);

		ValueArg<int> automatic("J", "automatic", "level of automatic search: 0,1,2 ", false, 0, "double");
		cmd.add(automatic);

		//END OF SEARCH Arguments

		ValueArg<std::string> out_file("O", "output_file", "Path of the output file", false, "BNN", "string");
		cmd.add(out_file);

		ValueArg<std::string> in_file("I", "input_file", "Path of the input file", false, "", "string");
		cmd.add(in_file);

		//
		// Parse the command line.
		//
		cmd.parse(argc,argv);
		//
		// Set variables
		//
    _sol_path = isolution.getValue();
    _data_path = idata.getValue();
		_index_model = imodel.getValue();
		_seed = seed.getValue();
		_nb_examples = nb_ex.getValue();
		_nb_examples_per_label = nb_ex_per_label.getValue();
		_k = param_k.getValue();
		_time = time.getValue();
		_prod_constraint = bool_prod.getValue();

		//Search parameters
		_search_strategy = search_strategy.getValue();
		_per_layer_branching = per_layer_branching.getValue();
		_variable_heuristic = variable_heuristic.getValue();
		_value_heuristic = value_heuristic.getValue();
		_automatic = automatic.getValue() ;

		//_strategy is no longer used
		if (!_search_strategy)
			_strategy = "default" ;
		else{
		//_strategy = std::to_string(_search_strategy);
		_strategy = std::to_string (_per_layer_branching) ;
		_strategy.append("-") ;
		_strategy.append(_variable_heuristic) ;
		_strategy.append("-") ;
		_strategy.append(_value_heuristic) ;
		_strategy.append("-") ;
		_strategy.append(std::to_string (_automatic)) ;
		}

		std::cout << " c _strategy is : " << _strategy << std::endl;

		_output_path = out_file.getValue();
		_input_file = in_file.getValue();
		_check_solution = check_sol.getValue();
		_print_solution = print_sol.getValue();
		_eval = eval.getValue();

		std::vector<int> v = archi.getValue();
		for ( int i = 0; static_cast<unsigned int>(i) < v.size(); i++ ){
			architecture.push_back(v[i]);
			_nb_neurons += v[i];
		}
	} catch ( ArgException& e )
	{ std::cout << "ERROR: " << e.error() << " " << e.argId() << std::endl; }
}
