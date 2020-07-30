
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

char _index_model;
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
bool _print_solution;
bool _eval;
std::string _input_file;

void parseOptions(int argc, char** argv);
int rand_a_b(int a, int b);
void print_vector(const std::vector<std::vector<std::vector<int>>> &vecteur);

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
	Data *bnn_data = new Data(architecture);
	int status;

	switch (_index_model) {
	case '1':
	{
		if (_input_file != "") {
			operations_research::sat::CPModel_MinWeight model(bnn_data, _prod_constraint, filename, _input_file);
			model.run(_time, _strategy);
			status = model.print_statistics(_check_solution, _strategy);
			weights = std::move(model.get_weights_solution());
		} else {
			if (_nb_examples == 0 && _nb_examples_per_label != 0){
				operations_research::sat::CPModel_MinWeight model(_nb_examples_per_label, bnn_data, _prod_constraint, filename);
				model.run(_time, _strategy);
				status = model.print_statistics(_check_solution, _strategy);
				weights = std::move(model.get_weights_solution());
			 }
			else{
				if (_nb_examples != 0 && _nb_examples_per_label == 0) {
					operations_research::sat::CPModel_MinWeight model(bnn_data, _nb_examples, _prod_constraint, filename);
					model.run(_time, _strategy);
					status = model.print_statistics(_check_solution, _strategy);
					weights = std::move(model.get_weights_solution());
				}
				else{
					std::cout << " c Invalid number of examples : default mode launched" << '\n';
					operations_research::sat::CPModel_MinWeight model(bnn_data, 1, _prod_constraint, filename);
					model.run(_time, _strategy);
					status = model.print_statistics(_check_solution, _strategy);
					weights = std::move(model.get_weights_solution());
				}
			}
		}
		break;
	}
	case '2':
	{
		if (_input_file != "") {
			operations_research::sat::CPModel_MaxClassification model(bnn_data, _prod_constraint, filename, _input_file);
			model.run(_time, _strategy);
			status = model.print_statistics(_check_solution, _strategy);
			weights = std::move(model.get_weights_solution());
			if (_print_solution)
				model.print_solution(model.get_response());
		} else {
			if (_nb_examples == 0 && _nb_examples_per_label != 0){
				operations_research::sat::CPModel_MaxClassification model(_nb_examples_per_label, bnn_data, _prod_constraint, filename);
				model.run(_time, _strategy);
				status = model.print_statistics(_check_solution, _strategy);
				weights = std::move(model.get_weights_solution());
				if (_print_solution)
					model.print_solution(model.get_response());
			}
			else{
				if (_nb_examples != 0 && _nb_examples_per_label == 0) {
					operations_research::sat::CPModel_MaxClassification model(bnn_data, _nb_examples, _prod_constraint, filename);
					model.run(_time, _strategy);
					status = model.print_statistics(_check_solution, _strategy);
					weights = std::move(model.get_weights_solution());
					if (_print_solution)
						model.print_solution(model.get_response());
				}
				else{
					std::cout << " c Invalid number of examples : default mode launched" << '\n';
					operations_research::sat::CPModel_MaxClassification model(bnn_data, 1, _prod_constraint, filename);
					model.run(_time, _strategy);
					status = model.print_statistics(_check_solution, _strategy);
					weights = std::move(model.get_weights_solution());
					if (_print_solution)
						model.print_solution(model.get_response());
				}
			}
		}
		break;
	}
	case '3':
	{
		if (_input_file != "") {
			operations_research::sat::CPModel_MaxClassification2 model(bnn_data, _prod_constraint, filename, _input_file);
			model.run(_time, _strategy);
			status = model.print_statistics(_check_solution, _strategy);
			weights = std::move(model.get_weights_solution());
			if (_print_solution)
				model.print_solution(model.get_response());
		} else {
			if (_nb_examples == 0 && _nb_examples_per_label != 0){
				operations_research::sat::CPModel_MaxClassification2 model(_nb_examples_per_label, bnn_data, _prod_constraint, filename);
				model.run(_time, _strategy);
				status = model.print_statistics(_check_solution, _strategy);
				weights = std::move(model.get_weights_solution());
				if (_print_solution)
					model.print_solution(model.get_response());
			}
			else{
				if (_nb_examples != 0 && _nb_examples_per_label == 0) {
					operations_research::sat::CPModel_MaxClassification2 model(bnn_data, _nb_examples, _prod_constraint, filename);
					model.run(_time, _strategy);
					status = model.print_statistics(_check_solution, _strategy);
					weights = std::move(model.get_weights_solution());
					if (_print_solution)
						model.print_solution(model.get_response());
				}
				else{
					std::cout << " c Invalid number of examples : default mode launched" << '\n';
					operations_research::sat::CPModel_MaxClassification2 model(bnn_data, 1, _prod_constraint, filename);
					model.run(_time, _strategy);
					status = model.print_statistics(_check_solution, _strategy);
					weights = std::move(model.get_weights_solution());
					if (_print_solution)
						model.print_solution(model.get_response());
				}
			}
		}
		break;
	}
	case '4':
	{
		operations_research::sat::CPModel_MaxSum model(bnn_data, _nb_examples, _prod_constraint, filename);
		std::cout<<std::endl<<std::endl;
		model.run(_time ,  _strategy) ;
		status = model.print_statistics(_check_solution, _strategy);
		weights = std::move(model.get_weights_solution());
		break;
	}
	case '5':
	{
		if (_input_file != "") {
			operations_research::sat::CPModel_Robust model(bnn_data, _prod_constraint, filename, _input_file);
			model.run(_time, _strategy);
			status = model.print_statistics(_check_solution, _strategy);
			weights = std::move(model.get_weights_solution());
		} else {
			if (_nb_examples == 0 && _nb_examples_per_label != 0){
				operations_research::sat::CPModel_Robust model(_nb_examples_per_label, bnn_data, _prod_constraint, filename, _k);
				model.run(_time, _strategy);
				status = model.print_statistics(_check_solution, _strategy);
				weights = std::move(model.get_weights_solution());
			}

			else{
				if (_nb_examples != 0 && _nb_examples_per_label == 0) {
					operations_research::sat::CPModel_Robust model(bnn_data, _nb_examples, _prod_constraint, filename, _k);
					model.run(_time, _strategy);
					status = model.print_statistics(_check_solution, _strategy);
					weights = std::move(model.get_weights_solution());
				}
				else{
					std::cout << " c Invalid number of examples : default mode launched" << '\n';
					operations_research::sat::CPModel_Robust model(bnn_data, 1, _prod_constraint, filename, _k);
					model.run(_time, _strategy);
					status = model.print_statistics(_check_solution, _strategy);
					weights = std::move(model.get_weights_solution());
				}
			}
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
		std::cout << "c starting evaluation..." << '\n';
		if(status == 2 || status == 4){
			Evaluation test(weights, bnn_data, filename);
			std::cout << " c Testing accuracy of the model with activation function : "<< '\n';
			accuracy_test = test.run_evaluation(true, true);
			std::cout << " c Training accuracy of the model with activation function : "<< '\n';
			accuracy_train = test.run_evaluation(false, true);
			std::cout << " c Testing accuracy of the model with all good metric : "<< '\n';
			accuracy_test_bis = test.run_evaluation(true, false);
			std::cout << " c Training accuracy of the model with all good metric : "<< '\n';
			accuracy_train_bis = test.run_evaluation(false, false);
		}

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
		results << "d TEST_ACCURACY " << accuracy_test << std::endl;
		results << "d TRAIN_ACCURACY " << accuracy_train << std::endl;
		results << "d TEST_ACCURACY_MAX " << accuracy_test_bis << std::endl;
		results << "d TRAIN_ACCURACY_MAX " << accuracy_train_bis << std::endl;
		results.close();
	}

	delete bnn_data;

	return EXIT_SUCCESS;
}

int rand_a_b(int a, int b){
	return rand()%((b-a)+1)+a;
}

void print_vector(const std::vector<std::vector<std::vector<int>>> &vecteur){
	int count_0 = 0;
	int count_1 = 0;
	int count_m1 = 0;
    for (const auto& i : vecteur)
			for (const auto& j : i)
				for(const auto& k : j){
					if (k == 0) {
						count_0++;
					}
					if (k == 1) {
						count_1++;
					}
					if (k == -1) {
						count_m1++;
					}
				}
	std::cout << "nb 0 : "<< count_0 << '\n';
	std::cout << "nb 1 : "<< count_1 << '\n';
	std::cout << "nb -1 : "<< count_m1 << '\n';
}

void parseOptions(int argc, char** argv){
	try {
		CmdLine cmd("BNN Parameters", ' ', "0.99" );
		//
		// Define arguments
		//
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

		SwitchArg print_sol("P","print", "indicates if the solution returned has to be printed", false);
		cmd.add(print_sol);

		SwitchArg eval("F","evaluation", "indicates if the evaluation on the testing and training sets has to be done", false);
		cmd.add(eval);

		ValueArg<std::string> search_strategy("D", "strategy", "The search strategy", false, "default", "string");
		cmd.add(search_strategy);

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
		_index_model = imodel.getValue();
		_seed = seed.getValue();
		_nb_examples = nb_ex.getValue();
		_nb_examples_per_label = nb_ex_per_label.getValue();
		_k = param_k.getValue();
		_time = time.getValue();
		_prod_constraint = bool_prod.getValue();
		_strategy =search_strategy.getValue();
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
