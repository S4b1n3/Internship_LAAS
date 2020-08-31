
#include "data.h"
#include "solution.h"
#include "new_cp_model.h"
//#include "cp_model.h"
//#include "cp_robust_model.h"
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
std::string _output_file;
std::string _output_path;
bool _check_solution;
int _print_solution;
std::string _input_file;
bool _reified_constraints;
bool _eval;
bool _weak_metric;
int __workers ;


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



void parseOptions(int argc, char** argv);
int rand_a_b(int a, int b);
void print_vector(const std::vector<std::vector<std::vector<int>>> &vecteur);

int main(int argc, char **argv) {


	architecture.push_back(784);
	parseOptions(argc, argv);
	architecture.push_back(10);

	srand(_seed);

	Data bnn_data(architecture);


	operations_research::sat::Search_parameters search (_search_strategy , _per_layer_branching , _variable_heuristic , _value_heuristic , _automatic) ;


    operations_research::sat::New_CP_Model model(bnn_data);
    if (_nb_examples != 0)
        model.set_data(1, _nb_examples);
    if (_nb_examples_per_label != 0)
        model.set_data(2,_nb_examples_per_label);
    if (_input_file != "")
        model.set_data(_input_file);
    model.set_simple_robustness(_k);
    model.set_prod_constraint(_prod_constraint);
		model.set_weak_metric(_weak_metric);
    model.set_optimization_problem(_index_model);
		model.set_workets(__workers) ;
    model.set_reified_constraints(_reified_constraints);
    model.set_output_stream(_output_file, _output_path, _input_file);
		model.run(_time, search);
		model.print_statistics(_check_solution, _eval, _strategy);
		if (_print_solution){
			model.print_solution(model.get_response(), _print_solution);
		}


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
		MultiArg<int> archi("A", "archi", "Architecture of the model", false, "int");
		cmd.add(archi);

		SwitchArg per_layer_branching("B", "per_layer", "branch per layer", false);
		cmd.add(per_layer_branching);

		SwitchArg bool_prod("C","product_constraints", "indicates the use of product constraints", false);
		cmd.add(bool_prod);

		SwitchArg search_strategy("D", "specific_strategy", "Use a specific search strategy", false);
		cmd.add(search_strategy);

		ValueArg<int> nb_ex_per_label("E", "nb_examples_per_label", "Number of examples per label", false, 0, "int");
		cmd.add(nb_ex_per_label);

		SwitchArg eval("F","evaluation", "indicates if the evaluation on the testing and training sets has to be done", false);
		cmd.add(eval);

		ValueArg<std::string> value_heuristic("G", "value_heuristic", "value heuristic: max, min, median, none ", false, "none", "string");
		cmd.add(value_heuristic);

		ValueArg<std::string> variable_heuristic("H", "var_heuristic", "variable heuristic: lex, min_domain, none", false, "none", "string");
		cmd.add(variable_heuristic);

		ValueArg<std::string> in_file("I", "input_file", "Path of the input file", false, "", "string");
		cmd.add(in_file);

		ValueArg<int> automatic("J", "automatic", "level of automatic search: 0,1,2 ", false, 0, "double");
		cmd.add(automatic);

		ValueArg<int> param_k("K", "k", "Robustness parameter", false, 0, "int");
		cmd.add(param_k);

		SwitchArg weak_metric("L", "weak_metric", "indicates the metric used to classify examples", false);
		cmd.add(weak_metric);

		ValueArg<char> imodel ("M", "index_model", "Index of the model to run : 0 for satisfaction, 1 for min weight and 2 for max classification", true, '1', "char");
		cmd.add(imodel);

		ValueArg<std::string> out_file("O", "output_file", "Name of the output file", false, "", "string");
		cmd.add(out_file);

		ValueArg<int> print_sol("P","print", "indicates if the solution returned has to be printed, and to which level : 0 nope, 1 minimal, 2 full", false, 0, "int");
		cmd.add(print_sol);

		SwitchArg reified_const("R", "reified", "Use of reified constraints", false);
		cmd.add(reified_const);

		ValueArg<int> seed ("S", "seed", "Seed", false, 1, "int");
		cmd.add(seed);

		ValueArg<double> time("T", "time", "Time limit for the solver", false, 1200.0, "double");
		cmd.add(time);

		ValueArg<std::string> out_path("U", "output_path", "Path of the output file", false, "BNN", "string");
		cmd.add(out_path);

		SwitchArg check_sol("V","check", "indicates if the solution returned has to be tested", false);
		cmd.add(check_sol);

		ValueArg<int> workers("W","workers", "number of workers", false, 0, "int");
		cmd.add(workers);

		ValueArg<int> nb_ex("X", "nb_examples", "Number of examples", false, 0, "int");
		cmd.add(nb_ex);


		//END OF SEARCH Arguments



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
    _output_file = out_file.getValue();
    _output_path = out_path.getValue();
    _input_file = in_file.getValue();
    _check_solution = check_sol.getValue();
    _print_solution = print_sol.getValue();
    _reified_constraints = reified_const.getValue();
    _eval = eval.getValue();
		_weak_metric = weak_metric.getValue();
		__workers = workers.getValue();


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

		std::vector<int> v = archi.getValue();
		for ( int i = 0; static_cast<unsigned int>(i) < v.size(); i++ ){
			architecture.push_back(v[i]);
			_nb_neurons += v[i];
		}
	} catch ( ArgException& e )
	{ std::cout << "ERROR: " << e.error() << " " << e.argId() << std::endl; }
}
