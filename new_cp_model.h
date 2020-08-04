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
        CpSolverResponse response;
        Domain domain;
        Domain activation_domain;

        int nb_examples;
        std::vector<std::vector<uint8_t>> inputs;
        std::vector<int> labels;
        std::vector<int> idx_examples;

        std::vector<std::vector<std::vector<int>>> weights_reference;
        bool check_model;

        bool prod_constraint;
        int optimization_problem;
        bool reified_constraints;
        int simple_robustness;

        //ORTools requires coefficients to be int64
        std::vector<std::vector<int64>> activation_first_layer;
        std::vector <std::vector<std::vector<IntVar>>> activation;
        std::vector <std::vector<std::vector<IntVar>>> preactivation;
        //weights[a][b][c] is the weight variable of the arc between neuron b on layer a-1 and neuron c on layer a
        std::vector<bool> weight_fixed_to_0;
        std::vector<std::vector <std::vector<IntVar>>> weights;
        std::vector<std::vector <std::vector<BoolVar>>> weight_is_0;



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

        New_CP_Model(Data _data):
            domain(-1,1),
            activation_domain(Domain::FromValues({-1,1}))
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
            check_model = false;
            switch (_type_data) {
                case 1:{
                    nb_examples = _nb_examples;
                    index_rand = rand()%(60000-_nb_examples);
                    for (size_t i = 0; i < nb_examples; ++i)
                        init_dataset(index_rand+i);
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
                                ++compt_ex;
                                ++occ[label];
                            }
                        }
                    }
                    break;
                }
            }
        }

        void set_data(const std::string &_input_file){
            nb_examples = 0;
            check_model = false;
            std::ifstream input_file(_input_file);
            std::vector<int> index_temp;
            if(input_file){
                std::string line;
                while (std::getline(input_file, line)){
                    if (line.substr(0, 8) == "INDEXES "){
                        std::string temp_line = line.substr(8);
                        std::vector<std::string> temp;
                        split(temp_line, temp, ' ');
                        for (size_t i = 0; i < temp.size(); ++i) {
                            index_temp.push_back(std::stoi(temp[i].c_str()));
                            ++nb_examples;
                        }
                    }
                }
            } else
                std::cout << "Error opening input file : " << _input_file << '\n';

            for(const int &i : index_temp)
                init_dataset(i);
        }

        void set_data(const std::string &_input_file, const std::string &_solution_file){
            nb_examples = 0;
            check_model = true;
            std::ifstream input_file(_input_file.c_str());
            std::vector<int> index_temp;
            if(input_file){
                std::string line;
                while (std::getline(input_file, line)){
                    if (line.substr(0, 8) == "INDEXES "){
                        std::string temp_line = line.substr(8);
                        std::vector<std::string> temp;
                        split(temp_line, temp, ' ');
                        for (size_t i = 0; i < temp.size(); ++i) {
                            index_temp.push_back(std::stoi(temp[i].c_str()));
                            ++nb_examples;
                        }
                    }
                }
            } else
                std::cout << "Error opening dataset file " << _input_file << '\n';

            for(const int &i : index_temp)
                init_dataset(i);

            std::ifstream solution_file(_solution_file.c_str());
            if (solution_file) {
                std::string line;
                std::vector<int> architecture;

                while (std::getline(solution_file, line)){
                    if (line.substr(0, 6) == "ARCHI ") {
                        std::string temp;
                        for (size_t i = 6; i < line.size(); ++i) {
                            if (line[i] != ' ')
                                temp += line[i];
                            if (line[i] == ' ') {
                                architecture.push_back(std::stoi(temp));
                                temp = "";
                            }
                        }
                    }
                    if (line.substr(0, 8) == "WEIGHTS ") {
                        int index_str = 8;
                        weights_reference.resize(architecture.size());
                        for (size_t l = 1; l < architecture.size(); ++l) {
                            weights_reference[l-1].resize(architecture[l-1]);
                            for (size_t i = 0; i < architecture[l-1]; ++i) {
                                weights_reference[l-1][i].resize(architecture[l]);
                                for (size_t j = 0; j < architecture[l]; ++j) {
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
            } else
                std::cout << "Error opening solution file : "<< _solution_file << '\n';

        }

        void set_data(const std::vector<std::vector<std::vector<int>>> &_weights, const std::vector<int> &_indexes_examples){
            nb_examples = _indexes_examples.size();
            check_model = true;
            weights_reference = _weights;
            for (const int &i : _indexes_examples)
                init_dataset(i);
        }

        void set_model_config(const int &_simple_robustness, const bool &_prod_constraint, const int &_optimization_problem, const bool &_reified_constraints){
            simple_robustness = _simple_robustness;
            prod_constraint = _prod_constraint;
            optimization_problem = _optimization_problem;
            reified_constraints = _reified_constraints;
            preactivation.resize(nb_examples);
            activation.resize(nb_examples);
            activation_first_layer.resize(nb_examples);
        }

        /* Getters */

        //returns the data of the problem
        Data get_data() const{
            return bnn_data;
        }

        //return the response of the problem
        CpSolverResponse get_response() const{
            return response;
        }

        int get_robustness() const {
            return simple_robustness;
        }

        /* get_a_lj method
        Parameters :
        - index_example : index of the example to classify
        - l : layer \in [1, bnn_data.get_layers()]
        - j : neuron on layer l \in [0, bnn_data.get_archi(l)]
        Output :
        a_{lj} variables from the CP paper
	    */
        IntVar get_a_lj(const int &index_example, const int &l, const int &j){
            assert(index_example>=0);
            assert(index_example<nb_examples);
            assert(l>0);
            assert(l<bnn_data.get_layers());
            assert(j>=0);
            assert(j<bnn_data.get_archi(l));
            return preactivation[index_example][l-1][j];
        }

        /* get_w_ilj method
        Parameters :
        - i : neuron on layer l-1 \in [0, bnn_data.get_archi(l-1)]
        - l : layer \in [1, bnn_data.get_layers()-1]
        - j : neuron on layer l \in [0, bnn_data.get_archi(l)]
        Output :
        w_{ilj} variables from the CP paper
	    */
        IntVar get_w_ilj(const int &i, const int &l, const int &j){

            assert(l>0);
            assert(l<bnn_data.get_layers());
            assert(i>=0);
            assert(i<bnn_data.get_archi(l-1));
            assert(j>=0);
            assert(j<bnn_data.get_archi(l));
            if ((l==1) && (nb_examples>1) )
                assert (! weight_fixed_to_0 [i]);
            return weights[l-1][i][j];
        }

        BoolVar get_weight_is_0_ilj(const int &i, const int &l, const int &j){
            assert(l>=2);
            assert(l<bnn_data.get_layers());
            assert(i>=0);
            assert(i<bnn_data.get_archi(l-1));
            assert(j>=0);
            assert(j<bnn_data.get_archi(l));
            if ((l==2) && (nb_examples>1) )
                assert (! weight_fixed_to_0 [i]);

            return weight_is_0[l-2][i][j];
        }

        /* declare_activation_variable method
        Parameters :
        - index_example : index of the training example to classify
        Output : None
        n_{lj} variables from the CP paper
	    */
        void declare_activation_variables(const int &index_example){

            int size = inputs[index_example].size();
            activation_first_layer[index_example].resize(size);
            for (size_t i = 0; i < size; ++i) {
                activation_first_layer[index_example][i] = (int)inputs[index_example][i];
                if (activation_first_layer[index_example][i] == 0) {
                    int size_second_layer = bnn_data.get_archi(1);
                    if(!check_model){
                        for (size_t j = 0; j < size_second_layer; ++j)
                            cp_model_builder.AddEquality(get_w_ilj(i, 1, j), 0);
                    }
                }
            }

            int number_remaining_layers = bnn_data.get_layers()-1;
            activation[index_example].resize(number_remaining_layers);
            for (size_t l = 0; l < number_remaining_layers; ++l) {
                int size_current_layer = bnn_data.get_archi(l+1);
                activation[index_example][l].resize(size_current_layer);
                for(size_t j = 0; j < size_current_layer; ++j){
                    activation[index_example][l][j] = cp_model_builder.NewIntVar(activation_domain);
                }
            }

        }

        /* declare_preactivation_variable method
        Parameters :
        - index_example : index of the training example to classify
        Output : None
        preactivation[l] represents the preactivation of layer l+1 where l \in [0,bnn_data.get_layers()-1]
        a_{lj} variables from the CP paper
	    */
        void declare_preactivation_variables(const int &index_example){

            int sum_image = 0 , sz = inputs[index_example].size();
            for(int i= 0; i <   sz; ++i)
            {
                if (! weight_fixed_to_0[i])
                    sum_image += (int) inputs[index_example][i]  ;
            }

            std::cout  << " c Test simple robustness with k = " << simple_robustness << std::endl;

            int number_layers = bnn_data.get_layers()-1;
            preactivation[index_example].resize(number_layers);
            for (size_t l = 0; l < number_layers; ++l) {
                int size_next_layer = bnn_data.get_archi(l+1);
                preactivation[index_example][l].resize(size_next_layer);
                int size_current_layer = bnn_data.get_archi(l);
                for(size_t j = 0; j < size_next_layer; ++j){
                    if(l == 0){
                        preactivation[index_example][l][j] = cp_model_builder.NewIntVar(Domain(-sum_image, sum_image));
                        if (simple_robustness > 0)
                        {
                            IntVar abs_preact = cp_model_builder.NewIntVar(Domain(simple_robustness,sum_image));
                            cp_model_builder.AddAbsEquality(abs_preact, preactivation[index_example][l][j]);
                        }
                    }
                    else {
                        preactivation[index_example][l][j] = cp_model_builder.NewIntVar(Domain(-size_current_layer, size_current_layer));
                    }
                }
            }
        }

        void declare_weight_variables() {

            //Initialisation of the variables
            int nb_layers = bnn_data.get_layers();
            weights.resize(nb_layers-1);
            //std::cout << " c weight is fixed size is " << bnn_data.get_archi(nb_layers-1) << std::endl;
            //We use weight_is_0 only for all layers except the first one (the pre-activation constraints from layer  0 et 0 use a linear constraint).
            if (prod_constraint)
                weight_is_0.resize(nb_layers-2);
            for (size_t l = 1; l < nb_layers; ++l) {
                int size_previous_layer = bnn_data.get_archi(l-1);
                weights[l-1].resize(size_previous_layer);
                if (prod_constraint && (l>=2))
                    weight_is_0[l-2].resize(size_previous_layer);

                for(size_t i = 0; i < size_previous_layer; ++i){
                    int size_current_layer = bnn_data.get_archi(l);
                    weights[l-1][i].resize(size_current_layer);
                    if (prod_constraint && (l>=2))
                        weight_is_0[l-2][i].resize(size_current_layer);
                    for (size_t j = 0; j < size_current_layer; ++j) {

                        /*One weight for each connection between the neurons i of layer
                      l-1 and the neuron j of layer l : N(i) * N(i+1) connections*/

                        weights[l-1][i][j] = cp_model_builder.NewIntVar(domain);
                        if(check_model){
                            cp_model_builder.AddEquality(weights[l-1][i][j], weights_reference[l-1][i][j]);
                        }
                        if (prod_constraint && (l>=2)){
                            weight_is_0[l-2][i][j] = cp_model_builder.NewBoolVar();

                            cp_model_builder.AddEquality(weights[l-1][i][j] , 0).OnlyEnforceIf(weight_is_0[l-2][i][j]);
                            cp_model_builder.AddNotEqual(weights[l-1][i][j] , 0).OnlyEnforceIf(Not(weight_is_0[l-2][i][j]));
                        }

                    }
                }
            }

            std::cout << " \n c START SETTING Weights to 0  \n  " << std::endl;

            int first_layer_size = bnn_data.get_archi(0) , count=0 ;
            weight_fixed_to_0.resize(first_layer_size) ;
            if (nb_examples> 1){

                int second_layer_size = bnn_data.get_archi(1);
                for (int i = 0 ; i< first_layer_size ; ++i){
                    int value_pixed = inputs[0][i] ;
                    bool fixed = true;
                    for (int idx = 1 ; idx< inputs.size() ; ++idx){
                        if ( inputs[idx][i] != value_pixed ){
                            fixed = false;
                            break;
                        }
                    }
                    if (fixed)
                    {
                        ++count;
                        weight_fixed_to_0.push_back(true);
                        for (int j = 0 ; j< second_layer_size ; ++j)
                            cp_model_builder.AddEquality(weights[0][i][j], 0);
                    }
                    else
                        weight_fixed_to_0.push_back(false);

                }
            }
            else
            {
                std::cout << " \n \n c SETTING Weights to 0 doesn't work with one training example \n" << std::endl;
                assert (false);
            }

            std::cout << " \n c " << count << " Weights on the first layer are fixed to 0" << std::endl;
            std::cout << " d FIRST_LAYER_FIXED_WEIGHTS " << count << std::endl;

        }

        void run(const double &nb_seconds, Search_parameters search) {
            assert(nb_seconds>0);
            std::cout << " c declare variables and constraints " << std::endl;
            std::clock_t c_start = std::clock();

            declare_weight_variables();
            for (size_t i = 0; i < nb_examples; i++) {
                declare_preactivation_variables(i);
                declare_activation_variables(i);
            }
            std::clock_t c_end = std::clock();

            std::cout << " c Setup finished; CPU setup time is " << (c_end-c_start) / CLOCKS_PER_SEC << " s" <<std::endl;
            std::cout<< " c running the solver.. " <<std::endl;
        }


    };

  }
}

#endif
