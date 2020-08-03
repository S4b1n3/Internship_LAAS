#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <cstdlib>
#include <dirent.h>
#include <cstring>
#include <sstream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <cinttypes>

#include "get_dataset.h"
#include "tclap/CmdLine.h"
using namespace TCLAP;

int _seed;
int _nb_examples;
int _index;
std::vector<int> architecture;

void parseOptions(int argc, char** argv);


int main(int argc, char **argv) {

    architecture.push_back(784);
  	parseOptions(argc, argv);
  	architecture.push_back(10);

    srand(_seed);
    const std::string path_folder("dataset");

    switch (_index) {
      case 1: {
        per_label(_nb_examples, path_folder, _seed);
        break;
      }
      case 2 : {
        random(_nb_examples, path_folder, _seed);
        break;
      }
      case 3 : {
        std::string solution_file = "solutions/solution_"+std::to_string(_seed);
        for (size_t i = 1; i < architecture.size()-1; i++)
          solution_file.append("_"+std::to_string(architecture[i]));
        solution_file.append(".sol");

        std::string data_file = path_folder+"/correct_"+std::to_string(_seed);
        for (size_t i = 1; i < architecture.size()-1; i++)
          data_file.append("_"+std::to_string(architecture[i]));
        data_file.append(".data");

        correct(data_file, solution_file, architecture);
        break;
      }
      default : std::cout << "Selected sampling mode is incorrect, please select 1 for \" random \" mode, 2 for \" per label \" mode and 3 for \" correct \" mode" << '\n';
    }

    return 0;
}

void parseOptions(int argc, char** argv){
	try {
		CmdLine cmd("BNN Checker Parameters", ' ', "0.99" );
		//
		// Define arguments
		//

		ValueArg<int> imode ("I", "index", "Index of the sampling mode", true, 1, "int");
		cmd.add(imode);

		ValueArg<int> seed ("S", "seed", "Seed", true, 1, "int");
		cmd.add(seed);

		ValueArg<int> nb_ex("X", "nb_examples", "Number of examples", false, 0, "int");
		cmd.add(nb_ex);

		MultiArg<int> archi("A", "archi", "Architecture of the model", false, "int");
		cmd.add(archi);

		//
		// Parse the command line.
		//
		cmd.parse(argc,argv);
		//
		// Set variables
		//
		_index = imode.getValue();
		_seed = seed.getValue();
		_nb_examples = nb_ex.getValue();

		std::vector<int> v = archi.getValue();
		for ( int i = 0; static_cast<unsigned int>(i) < v.size(); i++ ){
			architecture.push_back(v[i]);
		}
	} catch ( ArgException& e )
	{ std::cout << "ERROR: " << e.error() << " " << e.argId() << std::endl; }
}
