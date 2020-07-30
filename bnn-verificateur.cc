
#include "data.h"
#include "solution.h"
#include "cp_minweight_model.h"
#include "cp_maxclassification_model.h"
#include "cp_maxclassification_model2.h"
#include "cp_maxsum_weights_example_model.h"
#include "cp_robust_model.h"
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
std::string _strategy;
std::string _output_path;
bool _check_solution;
bool _print_solution;
bool _eval;
std::string _input_file;

std::vector<std::vector<std::vector<int>>> weights_temp;

void parseOptions(int argc, char** argv);
int rand_a_b(int a, int b);
bool fexists(const std::string& filename);


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

  if(!fexists(_sol_path)){
    std::cout << " c creating solution" << '\n';
    std::ofstream solution(_sol_path.c_str(), std::ios::out);

    solution << "ARCHI ";
    for (size_t i = 0; i < architecture.size(); i++) {
      solution << architecture[i] << " ";
    }
    solution << std::endl;
    solution << "WEIGHTS ";

    int tmp = architecture.size()-1;
  	weights_temp.resize(tmp);
  	for (size_t i = 1; i < tmp+1; i++) {
  		int tmp2 = architecture[i-1];
  		weights_temp[i-1].resize(tmp2);
  		for (size_t j = 0; j < tmp2; j++) {
  			int tmp3 = architecture[i];
  			weights_temp[i-1][j].resize(tmp3);
  			for (size_t k = 0; k < tmp3; k++) {
  				weights_temp[i-1][j][k] = rand_a_b(-1,1);
          solution << weights_temp[i-1][j][k] << " ";
  			}
  		}
  	}
  }


	filename.append("/results_"+_strategy+".stat");


	double accuracy_train, accuracy_test, accuracy_train_bis, accuracy_test_bis;

	std::vector<std::vector<std::vector<int>>> weights;
	Data *bnn_data = new Data(architecture);
	int status;

  switch (_index_model) {
    case '1':
    {
      if (_sol_path != "solution" && _data_path != "dataset") {
        if(!fexists(_data_path)){
            std::cout << " c creating dataset" << '\n';
            correct(_data_path, _sol_path);
        }
        operations_research::sat::CPModel_MinWeight model(bnn_data, _prod_constraint, filename, _data_path, _sol_path);
        model.run(_time, _strategy);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
      }
      if (_sol_path == "solution" && _data_path == "dataset" && _nb_examples == 0 && _nb_examples_per_label == 0) {
        std::vector<int> correct_examples;
        Evaluation test(weights_temp, bnn_data);
        correct_examples = test.get_correct_examples();
        operations_research::sat::CPModel_MinWeight model(bnn_data, _prod_constraint, filename, weights_temp, correct_examples);
        model.run(_time, _strategy);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
      }
      break;
    }
    case '2':
    {
      if (_sol_path != "solution" && _data_path != "dataset") {
        if(!fexists(_data_path)){
            std::cout << " c creating dataset" << '\n';
            correct(_data_path, _sol_path);
        }
        operations_research::sat::CPModel_MaxClassification model(bnn_data, _prod_constraint, filename, _data_path, _sol_path);
        model.run(_time, _strategy);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
      }
      if (_sol_path == "solution" && _data_path == "dataset" && _nb_examples == 0 && _nb_examples_per_label == 0) {
        std::vector<int> correct_examples;
        Evaluation test(weights_temp, bnn_data);
        correct_examples = test.get_correct_examples();
        operations_research::sat::CPModel_MaxClassification model(bnn_data, _prod_constraint, filename, weights_temp, correct_examples);
        model.run(_time, _strategy);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
      }
      break;
    }
    case '3':
    {
      if (_sol_path != "solution" && _data_path != "dataset") {
        if(!fexists(_data_path)){
            std::cout << " c creating dataset" << '\n';
            correct(_data_path, _sol_path);
        }
        operations_research::sat::CPModel_MaxClassification2 model(bnn_data, _prod_constraint, filename, _data_path, _sol_path);
        model.run(_time, _strategy);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
      }
      if (_sol_path == "solution" && _data_path == "dataset" && _nb_examples == 0 && _nb_examples_per_label == 0) {
        std::vector<int> correct_examples;
        Evaluation test(weights_temp, bnn_data);
        correct_examples = test.get_correct_examples();
        operations_research::sat::CPModel_MaxClassification2 model(bnn_data, _prod_constraint, filename, weights_temp, correct_examples);
        model.run(_time, _strategy);
        status = model.print_statistics(_check_solution, _strategy);
        weights = std::move(model.get_weights_solution());
      }
      break;
    }
    default:
  	{
  		std::cout << " c There is no model with index "<< _index_model << '\n';
  		std::cout << " c Please select 1, 2, 3 or 4, 5" << '\n';
  	}
  }

  delete bnn_data;
  return EXIT_SUCCESS;
}

int rand_a_b(int a, int b){
	return rand()%((b-a)+1)+a;
}

bool fexists(const std::string& filename) {
  std::ifstream ifile(filename.c_str());
  return (bool)ifile;
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
    _sol_path = isolution.getValue();
    _data_path = idata.getValue();
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
