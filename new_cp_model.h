#ifndef EXAMPLES_CPP_NEW_CP_MODEL_H
#define EXAMPLES_CPP_NEW_CP_MODEL_H

#include "ortools/sat/cp_model.h"
#include "ortools/sat/model.h"
#include "ortools/sat/sat_parameters.pb.h"

#include "ortools/util/sorted_interval_list.h"
#include "ortools/sat/cp_model_checker.h"

#include <memory>
#include "ortools/port/sysinfo.h"

#include <ctime>


#include <typeinfo>
#include <cmath>
#include <algorithm>
#include <memory>
#include <time.h>

#include <cstdio>
#include <cstdint>
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
#include <cinttypes>



namespace operations_research{
  namespace sat{

    //The different parameters of a search strategy
    class Search_parameters {

    public :
    	//Use a specific search strategy (i.e. non default)
    	bool _search_strategy ;
    	//Use layer per layer branching
    	bool _per_layer_branching ;
    	std::string _variable_heuristic ;
    	std::string	_value_heuristic ;
    	// Use automatic search ?
    	int _automatic ;
    	Search_parameters ( bool s, bool p, std::string var, std::string val, int a) :
    		_search_strategy(s), _per_layer_branching(p) ,  _variable_heuristic (var) , _value_heuristic (val), _automatic(a) {
    	}
    };

    class New_CP_Model {
    protected:

      Data bnn_data;
      CpModelBuilder cp_model_builder;

      int nb_examples;
      std::vector<std::vector<uint8_t>> inputs;
      std::vector<int> labels;
      std::vector<int> idx_examples;
      
      std::vector<std::vector<std::vector<int>>> weights_reference;


    public:

        //method used to split a string given a delimiter and put the result in a container
        template <class Container>
        void split(const std::string& str, Container& cont, char delim = ' ')
        {
            std::stringstream ss(str);
            std::string token;
            while (std::getline(ss, token, delim)) {
                cont.push_back(token);
            }
        }

        New_CP_Model(Data _data)
        {
            bnn_data = _data;
            std::cout << " c NUMBER OF LAYERS "<<bnn_data.get_layers() << '\n';
            bnn_data.print_archi();
            bnn_data.print_dataset();
        }

        void init_dataset(const int &index) {
            inputs.push_back(bnn_data.get_dataset().training_images[index]);
            labels.push_back((int)bnn_data.get_dataset().training_labels[index]);
            idx_examples.push_back(index);
        }

        void set_data(const int &_type_data, const int &_nb_examples){
            int index_rand;
            switch (_type_data) {
                case 1:{
                    nb_examples = _nb_examples;
                    index_rand = rand()%(60000-_nb_examples);
                    for (size_t i = 0; i < nb_examples; i++) {
                        init_dataset(index_rand+i);
                    }
                    std::cout << " c dataset size : " << inputs.size() << std::endl;
                    std::cout << " c label size : " << labels.size() << std::endl;
                    std::cout << " c indexes : ";
                    for (const auto &i : idx_examples)
                        std::cout << i << " ";

                    break;
                }
                case 2:{
                    nb_examples = 10*_nb_examples;
                    int compt_ex = 0;
                    std::vector<int> occ(10, 0);
                    std::vector<int> ind;
                    while (compt_ex < nb_examples) {
                        index_rand = rand()%60000;
                        auto it = std::find(std::begin(ind), std::end(ind), index_rand);
                        if (it == ind.end()) {
                            ind.push_back(index_rand);
                            int label = (int)bnn_data.get_dataset().training_labels[index_rand];
                            if(occ[label] < _nb_examples){
                                init_dataset(index_rand);
                                compt_ex++;
                                occ[label]++;
                            }
                        }
                    }
                    std::cout << " c dataset size : " << inputs.size() << std::endl;
                    std::cout << " c label size : " << labels.size() << std::endl;
                    std::cout << " c indexes : ";
                    for (const auto &i : idx_examples)
                        std::cout << i << " ";
                    break;
                }
            }
        }

        void set_data(const std::string &_input_file){
            nb_examples = 0;
            std::ifstream input_file(_input_file);
            std::vector<int> index_temp;
            if(input_file){
                std::string line;
                while (std::getline(input_file, line)){
                    if (line.substr(0, 8) == "INDEXES "){
                        std::string temp_line = line.substr(8);
                        std::vector<std::string> temp;
                        split(temp_line, temp, ' ');
                        for (size_t i = 0; i < temp.size(); i++) {
                            index_temp.push_back(std::stoi(temp[i].c_str()));
                            nb_examples++;
                        }
                    }
                }
            } else {
                std::cout << "Error opening input file : " << _input_file << '\n';
            }

            for(const int &i : index_temp){
                init_dataset(i);
            }
            std::cout << " c dataset size : " << inputs.size() << std::endl;
            std::cout << " c label size : " << labels.size() << std::endl;
            std::cout << " c indexes : ";
            for (const auto &i : idx_examples)
                std::cout << i << " ";
        }

        void set_data(const std::string &_input_file, const std::string &_solution_file){
            nb_examples = 0;
            std::ifstream input_file(_input_file.c_str());
            std::vector<int> index_temp;
            if(input_file){
                std::string line;
                while (std::getline(input_file, line)){
                    if (line.substr(0, 8) == "INDEXES "){
                        std::string temp_line = line.substr(8);
                        std::vector<std::string> temp;
                        split(temp_line, temp, ' ');
                        for (size_t i = 0; i < temp.size(); i++) {
                            index_temp.push_back(std::stoi(temp[i].c_str()));
                            nb_examples++;
                        }
                    }
                }
            } else {
                std::cout << "Error opening dataset file " << _input_file << '\n';
            }

            for(const int &i : index_temp){
                init_dataset(i);
            }

            std::ifstream solution_file(_solution_file.c_str());
            if (solution_file) {
                std::string line;
                std::vector<int> architecture;

                while (std::getline(solution_file, line)){
                    if (line.substr(0, 6) == "ARCHI ") {
                        std::string temp;
                        for (size_t i = 6; i < line.size(); i++) {
                            if (line[i] != ' ') {
                                temp += line[i];
                            }
                            if (line[i] == ' ') {
                                architecture.push_back(std::stoi(temp));
                                temp = "";
                            }
                        }
                    }
                    if (line.substr(0, 8) == "WEIGHTS ") {
                        int index_str = 8;
                        weights_reference.resize(architecture.size());
                        for (size_t l = 1; l < architecture.size(); l++) {
                            weights_reference[l-1].resize(architecture[l-1]);
                            for (size_t i = 0; i < architecture[l-1]; i++) {
                                weights_reference[l-1][i].resize(architecture[l]);
                                for (size_t j = 0; j < architecture[l]; j++) {
                                    if (line[index_str] == '-') {
                                        weights_reference[l-1][i][j] = -1;
                                        index_str += 3;
                                    }
                                    else {
                                        weights_reference[l-1][i][j] = line[index_str] - '0';
                                        index_str += 2;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                std::cout << "Error opening solution file : "<< _solution_file << '\n';
            }

            std::cout << " c dataset size : " << inputs.size() << std::endl;
            std::cout << " c label size : " << labels.size() << std::endl;
            std::cout << " c indexes : ";
            for (const auto &i : idx_examples)
                std::cout << i << " ";

        }




    };

  }
}

#endif
