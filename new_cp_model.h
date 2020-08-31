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

#include "evaluation.h"



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
        LinearExpr objective;
        SatParameters parameters;
        Model model;

        int nb_examples;
        std::vector<std::vector<uint8_t>> inputs;
        std::vector<int> labels;
        std::vector<int> idx_examples;

        std::vector<std::vector<std::vector<int>>> weights_reference;
        bool check_model;

        std::string output_file;
        bool prod_constraint;
        bool weak_metric;
        //0 for satisfaction, 1 for min_weight and 2 for max_classification
        char optimization_problem;
        bool reified_constraints;
        int simple_robustness;
        int decision_variables_size;
        std::ostream* out = &std::cout;
        std::ofstream fout;

        std::vector<BoolVar> classification;
        //ORTools requires coefficients to be int64
        std::vector<std::vector<int64>> activation_first_layer;
        std::vector <std::vector<std::vector<IntVar>>> activation;
        std::vector <std::vector<std::vector<IntVar>>> preactivation;
        //weights[a][b][c] is the weight variable of the arc between neuron b on layer a-1 and neuron c on layer a
        std::vector<bool> weight_fixed_to_0;
        std::vector<std::vector <std::vector<IntVar>>> weights;
        std::vector<std::vector <std::vector<BoolVar>>> weight_is_0;

        std::vector<std::vector <std::vector<int>>> weights_solution;
        std::vector <std::vector<int>> activation_solution;
        std::vector <std::vector<int>> preactivation_solution;
        std::vector<int> classification_solution;



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
            *out << " c NUMBER OF LAYERS "<<bnn_data.get_layers() << '\n';
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
                *out << " c Error opening input file : " << _input_file << '\n';

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
                *out << " c Error opening dataset file " << _input_file << '\n';

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
                    if (architecture != bnn_data.get_archi()){
                        *out << " c The architecture of the solution file is different from the architecture of the model. Please select an other file." << std::endl;
                        exit(EXIT_FAILURE);
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
                *out << " c Error opening solution file : "<< _solution_file << '\n';

        }

        void set_data(const std::vector<std::vector<std::vector<int>>> &_weights, const std::vector<int> &_indexes_examples){
            nb_examples = _indexes_examples.size();
            check_model = true;
            weights_reference = _weights;
            for (const int &i : _indexes_examples)
                init_dataset(i);
        }

        void set_simple_robustness(int robustness){
            this->simple_robustness = robustness;
        }

        void set_prod_constraint(bool use_product_constraints){
            this->prod_constraint = use_product_constraints;
        }

        void set_weak_metric(bool use_weak_metric){
          this->weak_metric = use_weak_metric;
        }

        void set_optimization_problem(char optimization_mode){
            this->optimization_problem = optimization_mode;
        }

        void set_reified_constraints(bool use_reified_constraints){
            this->reified_constraints = use_reified_constraints;
        }

        void set_workets(int w){
            parameters.set_num_search_workers(w);
        }

        void set_output_stream(std::string _output_file, std::string _output_path){
            if (_output_file != ""){
                create_result_file(_output_path, _output_file);
                fout.open(output_file, std::ios::app);
                out = &fout;
            }
        }

        void set_output_stream(std::string _output_file, std::string _output_path, std::string _input_file){
            if (_output_file != ""){
                create_result_file(_output_path, _output_file, _input_file);
                fout.open(output_file, std::ios::app);
                out = &fout;
            }
        }


        void set_model_config(){
            //Initialisation of the variables
            int nb_layers = bnn_data.get_layers();
            weights.resize(nb_layers-1);
            preactivation.resize(nb_examples);
            activation.resize(nb_examples);
            activation_first_layer.resize(nb_examples);

            weights_solution.resize(nb_layers-1);
            preactivation_solution.resize(nb_layers-1);
            activation_solution.resize(nb_layers);

            if (optimization_problem=='2'){
                classification.resize(nb_examples);
                classification_solution.resize(nb_examples);
            }
            if ((optimization_problem == '1' || optimization_problem == '0') && reified_constraints){
                *out << " c Reified contraints can not be used with min weight problem " << std::endl;
                *out << " c Changing boolean reified_constraint to false " << std::endl;
                reified_constraints = false;
            }
        }

        void set_porto_heuristic( DecisionStrategyProto* proto , std::string variable_heuristic,
                                  std::string value_heuristic){
            if (variable_heuristic == "lex")
                proto->set_variable_selection_strategy(DecisionStrategyProto::CHOOSE_FIRST);
            else if  (variable_heuristic == "min_domain")
                proto->set_variable_selection_strategy(DecisionStrategyProto::CHOOSE_MIN_DOMAIN_SIZE);
            else
                assert (variable_heuristic == "none") ;

            if (value_heuristic == "max")
                proto->set_domain_reduction_strategy(DecisionStrategyProto::SELECT_MAX_VALUE);
            else if  (value_heuristic == "min")
                proto->set_domain_reduction_strategy(DecisionStrategyProto::SELECT_MIN_VALUE);
            else if  (value_heuristic == "median")
                proto->set_domain_reduction_strategy(DecisionStrategyProto::SELECT_MEDIAN_VALUE);
            else
                assert (value_heuristic == "none");
        }

        void setup_branching( bool branch_per_layer,  std::string variable_heuristic,
                              std::string value_heuristic, int automatic ){
            *out << " c setting specific branching :  "
                      << " branch_per_layer : "
                      << " var : " << variable_heuristic
                      << " val : " << value_heuristic
                      << " automatic " << automatic <<	std::endl;
            decision_variables_size = cp_model_builder.Build().variables_size() ;

            std::vector<IntVar> w;
            std::vector<std::vector<IntVar>> w_level;

            if (branch_per_layer){
                int nb_layers = bnn_data.get_layers();
                w_level.resize(nb_layers -1) ;
                int count  = 0 ;
                for (size_t l = 1; l < nb_layers; l++){
                    int current  = bnn_data.get_archi(l);
                    int previous = bnn_data.get_archi(l-1);
                    for(size_t i = 0; i < previous; i++){
                        if ( ((l ==1) && !weight_fixed_to_0[i]) || (l> 1))
                            for (size_t j = 0; j < current; j++){
                                //Todo :  later removed fixed weights
                                w_level[l-1].push_back( weights[l-1][i][j] );
                                ++count;
                            }
                    }
                }
                decision_variables_size = count ;
            }
            else{
                int nb_layers = bnn_data.get_layers();
                for (size_t l = 1; l < nb_layers; l++){

                    int current = bnn_data.get_archi(l);
                    int previous = bnn_data.get_archi(l-1);
                    for(size_t i = 0; i < previous; i++){
                        if ( ((l ==1) && !weight_fixed_to_0[i]) || (l> 1))
                            for (size_t j = 0; j < current; j++){
                                //Todo :  later removed fixed weights
                                w.push_back( weights[l-1][i][j] );
                            }
                    }
                }
                decision_variables_size = w.size();
            }
            //std::vector<IntVar> reverse_w(w);
            //std::reverse(std::begin(reverse_w), std::end(reverse_w));
            *out << " c number of branching variables is "  << decision_variables_size  << std::endl;
            CpModelProto* cp_model_ = cp_model_builder.MutableProto() ;
            if (branch_per_layer){
                //Branch per layer
                for (int i =0; i < w_level.size() ; ++i){
                    DecisionStrategyProto* proto ;
                    proto = cp_model_->add_search_strategy();
                    for (const IntVar& var : w_level[i]) {
                        proto->add_variables(var.index());
                    }
                    set_porto_heuristic(proto, variable_heuristic, value_heuristic);
                }
            }
            else {
                //Branch on all weight variables
                DecisionStrategyProto* proto ;
                proto = cp_model_->add_search_strategy();
                for (const IntVar& var : w) {
                    proto->add_variables(var.index());
                }
                set_porto_heuristic(proto, variable_heuristic, value_heuristic);
            }

            if (automatic >0)
                parameters.set_search_branching(SatParameters::AUTOMATIC_SEARCH );
            if (automatic >1)
            {
                parameters.set_interleave_search(true);
                parameters.set_reduce_memory_usage_in_interleave_mode(true);
            }
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

        std::string get_output_file() const{
            return output_file;
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

            //*out  << " c Test simple robustness with k = " << simple_robustness << std::endl;

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

            //*out << " c weight is fixed size is " << bnn_data.get_archi(nb_layers-1) << std::endl;
            //We use weight_is_0 only for all layers except the first one (the pre-activation constraints from layer  0 et 0 use a linear constraint).
            int nb_layers = bnn_data.get_layers();
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

            *out << " \n c START SETTING Weights to 0  \n  " << std::endl;

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
                *out << " \n \n c SETTING Weights to 0 doesn't work with one training example \n" << std::endl;
                assert (false);
            }

            *out << " \n c " << count << " Weights on the first layer are fixed to 0" << std::endl;
            *out << " d FIRST_LAYER_FIXED_WEIGHTS " << count << std::endl;

        }

        void declare_classification_variable(){
            for (size_t i = 0; i < nb_examples; ++i) {
                classification[i] = cp_model_builder.NewBoolVar();
            }
        }

        void create_result_file(std::string _output_path, std::string filename){
            output_file = _output_path;

            output_file.append("/results/resultsM"+std::to_string(optimization_problem));

            if(prod_constraint)
                output_file.append("-C");

            if(reified_constraints)
                output_file.append("-R");


            output_file.append("/results");

            int nb_layers_minus_one = bnn_data.get_layers()-1;
            for (size_t i = 1; i < nb_layers_minus_one; ++i) {
                output_file.append("_"+std::to_string(bnn_data.get_archi(i)));
            }
            if (nb_layers_minus_one-1 == 0) {
                output_file.append("_0");
            }
            std::string cmd = "mkdir -p "+output_file;
            int launch_cmd = system(cmd.c_str());

            output_file.append("/"+filename);
        }

        void create_result_file(std::string _output_path, std::string filename, std::string _input_file){
            output_file = _output_path;

            int index = _input_file.find_last_of("/");
            std::string input_filename = _input_file.substr(index+1);


            output_file.append("/results/results_"+input_filename.substr(10, 1));

            output_file.append("/results_"+input_filename.substr(12, input_filename.size()-17));

            std::string cmd = "mkdir -p "+output_file;
            int launch_cmd = system(cmd.c_str());

            output_file.append("/"+filename);
        }

        /* model_objective_maximize_classification method
          This function sums all the values of classification in the LinearExpr objectif
          Parameters : None
          Output : None
        */
        void model_declare_objective(){
            if (optimization_problem == '1' || optimization_problem == '0'){
                //min_weight mode
                int nb_layers = bnn_data.get_layers();
                for (size_t l = 1; l < nb_layers; l++) {
                    int size_previous_layer = bnn_data.get_archi(l - 1);
                    for (size_t i = 0; i < size_previous_layer; i++) {
                        int size_current_layer = bnn_data.get_archi(l);
                        for (size_t j = 0; j < size_current_layer; j++) {
                            IntVar abs = cp_model_builder.NewIntVar(Domain(0, 1));
                            cp_model_builder.AddAbsEquality(abs, weights[l - 1][i][j]);
                            objective.AddVar(abs);
                        }
                    }
                }
            }
            if (optimization_problem == '2') {
                //max_classification mode
                for (size_t i = 0; i < nb_examples; i++)
                    objective.AddVar(classification[i]);
            }
        }


        /* model_activation_constraint method
            Parameters :
            - index_example : index of the example to classifie
            - l : layer \in [1, bnn_data.get_layers()]
            - j : neuron on layer l \in [0, bnn_data.get_archi(l)]

            preactivation[l][j] >= 0 => activation[l][j] = 1
            preactivation[l][j] < 0 => activation[l][j] = -1
            Output : None
    	 */
        void model_activation_constraints(const int &index_example, const int &l, const int &j){
            //_temp_bool is true iff preactivation[l][j] < 0
            //_temp_bool is false iff preactivation[l][j] >= 0

            BoolVar _temp_bool = cp_model_builder.NewBoolVar();

            if (reified_constraints) {
                cp_model_builder.AddLessThan(get_a_lj(index_example, l, j), 0).OnlyEnforceIf(
                        {_temp_bool, classification[index_example]});
                cp_model_builder.AddGreaterOrEqual(get_a_lj(index_example, l, j), 0).OnlyEnforceIf(
                        {Not(_temp_bool), classification[index_example]});
                cp_model_builder.AddEquality(activation[index_example][l - 1][j], -1).OnlyEnforceIf(
                        {_temp_bool, classification[index_example]});
                cp_model_builder.AddEquality(activation[index_example][l - 1][j], 1).OnlyEnforceIf(
                        {Not(_temp_bool), classification[index_example]});
            } else {
                cp_model_builder.AddLessThan(get_a_lj(index_example, l, j), 0).OnlyEnforceIf(_temp_bool);
                cp_model_builder.AddGreaterOrEqual(get_a_lj(index_example, l, j), 0).OnlyEnforceIf(Not(_temp_bool));
                cp_model_builder.AddEquality(activation[index_example][l - 1][j], -1).OnlyEnforceIf(_temp_bool);
                cp_model_builder.AddEquality(activation[index_example][l - 1][j], 1).OnlyEnforceIf(Not(_temp_bool));
            }

        }

        void model_preactivation_constraints(const int &index_example, const int &l, const int &j){

            if(l == 1){
                LinearExpr temp(0);
                int size_first_layer = bnn_data.get_archi(0);
                for (size_t i = 0; i < size_first_layer; ++i) {
                    if (activation_first_layer[index_example][i] != 0)
                        if (! weight_fixed_to_0[i])
                        {
                            temp.AddTerm(get_w_ilj(i, l, j), activation_first_layer[index_example][i]);
                        }
                }
                if (reified_constraints)
                    cp_model_builder.AddEquality(get_a_lj(index_example, 1, j), temp).OnlyEnforceIf(classification[index_example]);
                else
                    cp_model_builder.AddEquality(get_a_lj(index_example, 1, j), temp);
            }
            else{
                std::vector<IntVar> temp(bnn_data.get_archi(l-1));
                int size_previous_layer = bnn_data.get_archi(l-1) ;
                for (size_t i = 0; i < size_previous_layer; ++i) {
                    temp[i] = cp_model_builder.NewIntVar(domain);
                    if(!prod_constraint){

                        IntVar sum_weights_activation = cp_model_builder.NewIntVar(Domain(-2,2));
                        IntVar sum_temp_1 = cp_model_builder.NewIntVar(Domain(0, 2));
                        if (reified_constraints){
                          cp_model_builder.AddEquality(sum_weights_activation, LinearExpr::Sum({get_w_ilj(i, l, j), activation[index_example][l-2][i]})).OnlyEnforceIf(classification[index_example]);
                          cp_model_builder.AddEquality(sum_temp_1, temp[i].AddConstant(1)).OnlyEnforceIf(classification[index_example]);
                          cp_model_builder.AddAbsEquality(sum_temp_1, sum_weights_activation);// If we add .OnlyEnforceIf(classification[index_example]); with AbsEquality constraint, ORTools indicates "Enforcement literal not supported in constraint: enforcement_literal". It looks like AbsEquality can not be reified

                        }else{
                            cp_model_builder.AddEquality(sum_weights_activation, LinearExpr::Sum({get_w_ilj(i, l, j), activation[index_example][l-2][i]}));
                            cp_model_builder.AddEquality(sum_temp_1, temp[i].AddConstant(1));
                            cp_model_builder.AddAbsEquality(sum_temp_1, sum_weights_activation);
                        }
                    }
                    else {

                        /*
                      (C == 0) ssi (weights == 0)
                        (C == 0) => (weights == 0) et (weights == 0) => (C == 0)
                        Not(weights == 0) => Not(C == 0) et Not(C == 0) => (Not weights == 0)
                      (C == 1) ssi (a == b)
                        (C == 1) => (a == b) et (a == b) => (C == 1)
                        Not(a == b) => Not(C == 1) et Not(C == 1) => Not(a == b)

                         */

                        //BoolVar b1 = cp_model_builder.NewBoolVar();
                        //*out << " HERE " << std::endl;
                        //Implement b1 == (temp[i] == 0)
                        BoolVar b3 = cp_model_builder.NewBoolVar();

                        if (reified_constraints){
                            cp_model_builder.AddEquality(temp[i], 0).OnlyEnforceIf({get_weight_is_0_ilj(i,l,j), classification[index_example]});
                            cp_model_builder.AddNotEqual(temp[i], 0).OnlyEnforceIf({Not(get_weight_is_0_ilj(i,l,j)), classification[index_example]});
                            // Implement b3 == (temp[i] == 1)
                            cp_model_builder.AddEquality(temp[i], 1).OnlyEnforceIf({b3, classification[index_example]});
                            cp_model_builder.AddNotEqual(temp[i], 1).OnlyEnforceIf({Not(b3), classification[index_example]});
                            //Implement b3 == (weights == activation)
                            cp_model_builder.AddEquality(get_w_ilj(i, l, j), activation[index_example][l-2][i]).OnlyEnforceIf({b3, classification[index_example]});
                            cp_model_builder.AddNotEqual(get_w_ilj(i, l, j), activation[index_example][l-2][i]).OnlyEnforceIf({Not(b3), classification[index_example]});

                        }else{
                            cp_model_builder.AddEquality(temp[i], 0).OnlyEnforceIf(get_weight_is_0_ilj (i,l,j));
                            cp_model_builder.AddNotEqual(temp[i], 0).OnlyEnforceIf(Not(get_weight_is_0_ilj (i,l,j) ) );
                            // Implement b3 == (temp[i] == 1)
                            cp_model_builder.AddEquality(temp[i], 1).OnlyEnforceIf(b3);
                            cp_model_builder.AddNotEqual(temp[i], 1).OnlyEnforceIf(Not(b3));
                            //Implement b3 == (weights == activation)
                            cp_model_builder.AddEquality(get_w_ilj(i, l, j), activation[index_example][l-2][i]).OnlyEnforceIf(b3);
                            cp_model_builder.AddNotEqual(get_w_ilj(i, l, j), activation[index_example][l-2][i]).OnlyEnforceIf(Not(b3));

                        }
                    }
                }
                if (reified_constraints)
                    cp_model_builder.AddEquality(get_a_lj(index_example, l, j), LinearExpr::Sum(temp)).OnlyEnforceIf(classification[index_example]);
                else
                    cp_model_builder.AddEquality(get_a_lj(index_example, l, j), LinearExpr::Sum(temp));
            }
        }

        /* model_output_constraint method
          This function forces the output to match the label
          Parameters :
          - index_example : index of examples
          Output : None
        */
        void model_output_constraints(const int &index_example){
            int label = labels[index_example];
            int last_layer = bnn_data.get_layers()-2;
            int size_last_layer = bnn_data.get_archi(bnn_data.get_layers()-1);

            IntVar max_value = cp_model_builder.NewIntVar(Domain(-size_last_layer, size_last_layer));
            cp_model_builder.AddMaxEquality(max_value, preactivation[index_example][last_layer]);

            if (optimization_problem == '1' || optimization_problem == '0'){
                if (weak_metric) {
                  cp_model_builder.AddEquality(preactivation[index_example][last_layer][label], max_value);
                }
                else {
                  cp_model_builder.AddEquality(activation[index_example][last_layer][label], 1);
                  for (size_t i = 0; i < size_last_layer; i++) {
                      if (i != label) {
                          cp_model_builder.AddEquality(activation[index_example][last_layer][i], -1);
                      }
                  }
                }

            }

            if (optimization_problem == '2'){
              if (weak_metric) {
                cp_model_builder.AddEquality(preactivation[index_example][last_layer][label], max_value).OnlyEnforceIf(classification[index_example]);
              }
              else {
                LinearExpr last_layer_sum(0);
                for (size_t i = 0; i < size_last_layer; i++) {
                    if (i != label) {
                        last_layer_sum.AddVar(activation[index_example][last_layer][i]);
                    }
                }
                cp_model_builder.AddEquality(activation[index_example][last_layer][label], 1).OnlyEnforceIf(classification[index_example]);
                cp_model_builder.AddEquality(last_layer_sum, -(size_last_layer - 1)).OnlyEnforceIf(classification[index_example]);
              }
            }
        }

        /* run method
          This function calls all the necessary methods to run the solver
          Parameters :
          - nb_seconds : Sets a time limit of nb_seconds
          - search : search strategy of the solver
          Output : None
        */
        void run(const double &nb_seconds, Search_parameters search) {
            assert(nb_seconds > 0);
            *out << " c declare variables and constraints " << std::endl;
            std::clock_t c_start = std::clock();

            set_model_config();
            declare_weight_variables();
            if (optimization_problem=='2')
                declare_classification_variable();
            for (size_t i = 0; i < nb_examples; ++i) {
                declare_preactivation_variables(i);
                declare_activation_variables(i);
            }
            for (size_t i = 0; i < nb_examples; ++i) {
                int nb_layers = bnn_data.get_layers();
                for (size_t l = 1; l < nb_layers; ++l) {
                    int size_current_layer =  bnn_data.get_archi(l);
                    for (size_t j = 0; j < size_current_layer; ++j) {
                        model_preactivation_constraints(i, l, j);
                        model_activation_constraints(i, l, j);
                    }
                }
            }
            for (size_t i = 0; i < nb_examples; ++i) {
                model_output_constraints(i);
            }
            model_declare_objective();
            decision_variables_size = cp_model_builder.Build().variables_size() ;
            if (search._search_strategy)
                setup_branching(search._per_layer_branching, search._variable_heuristic,
                                search._value_heuristic, search._automatic);
            parameters.set_max_time_in_seconds(nb_seconds);     //Add a timelimit
            parameters.set_random_seed(1000);
            model.Add(NewSatParameters(parameters));
            std::clock_t c_end = std::clock();

            *out << " c Setup finished; CPU setup time is " << (c_end - c_start) / CLOCKS_PER_SEC << " s"
                      << std::endl;

            *out << "d SETUP TIME " << (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
            *out << " c Setup finished; CPU setup time is "  << (c_end-c_start) / CLOCKS_PER_SEC << " s" <<std::endl;
            *out << "\n c Some statistics on the model : " << std::endl;
            *out <<  "d VARIABLES " << cp_model_builder.Build().variables_size() << std::endl ;
            *out <<  "d DECISION_VARIABLES " << decision_variables_size << std::endl ;
            *out <<  "d CONSTRAINTS " << cp_model_builder.Build().constraints_size() << std::endl ;

            *out << " c running the solver.. " << std::endl;

            switch (optimization_problem) {
                case '1': {
                    cp_model_builder.Minimize(objective);
                    break;
                }
                case '2': {
                    cp_model_builder.Maximize(objective);
                    break;
                }
            }

            response = SolveCpModel(cp_model_builder.Build(), &model);

        }

        void check(const CpSolverResponse &r, const bool &check_sol, const std::string &strategy) {
            *out << " c entering check method " << std::endl;
            int nb_layers = bnn_data.get_layers();
            for (size_t l = 1; l < nb_layers; ++l) {
                int size_previous_layer = bnn_data.get_archi(l-1);
                weights_solution[l-1].resize(size_previous_layer);
                for (size_t i = 0; i < size_previous_layer; ++i) {
                    int size_current_layer = bnn_data.get_archi(l);
                    weights_solution[l-1][i].resize(size_current_layer);
                    for (size_t j = 0; j < size_current_layer; ++j)
                        weights_solution[l-1][i][j] = SolutionIntegerValue(r, weights[l-1][i][j]);
                }
            }

            int check_count = nb_examples;
            for (size_t i = 0; i < nb_examples; i++) {
                bool classif = true;

                if (optimization_problem == '2'){
                    classification_solution[i] = SolutionIntegerValue(r, classification[i]);
                    classif = (classification_solution[i] == 1);
                }


                for (size_t l = 0; l < nb_layers - 1; l++) {
                    int size_next_layer = bnn_data.get_archi(l + 1);
                    preactivation_solution[l].resize(size_next_layer);
                    for (size_t j = 0; j < size_next_layer; j++) {
                        preactivation_solution[l][j] = SolutionIntegerValue(r, preactivation[i][l][j]);
                    }
                }

                for (size_t l = 0; l < nb_layers; l++) {
                    int size_current_layer = bnn_data.get_archi(l);
                    activation_solution[l].resize(size_current_layer);
                    for (size_t j = 0; j < size_current_layer; j++) {
                        if (l == 0) {
                            activation_solution[l][j] = (int) activation_first_layer[i][j];
                        } else {
                            activation_solution[l][j] = SolutionIntegerValue(r, activation[i][l - 1][j]);
                        }
                    }
                }


                if (check_sol &&  (!reified_constraints || (reified_constraints && classif))) {
                    Solution check_solution(bnn_data, weights_solution, activation_solution, preactivation_solution);
                    if (!check_model) {
                        check_solution.set_evaluation_config(true, true, classif, !weak_metric, false);
                        *out << "d CHECKING_EXAMPLE "<<i << " : ";
                    }else
                        check_solution.set_evaluation_config(true, false, classif, !weak_metric, false);
                    bool checking = check_solution.run_solution_light(idx_examples[i]);
                    if (!checking) {
                        check_count--;
                    }
                }

            }
            *out << "d CHECKING "<<(check_count == nb_examples)<<std::endl;
        }

        // Print some statistics from the solver: Runtime, number of nodes, number of propagation (filtering, pruning), memory,
        // Status: Optimal, suboptimal, satisfiable, unsatisfiable, unkown
        // Output Status: {OPTIMAL, FEASIBLE, INFEASIBLE, MODEL_INVALID, UNKNOWN}
        void print_statistics(const bool &check_solution, const bool &eval, const std::string &strategy){
            *out << "\n c Some statistics on the solver response : " << '\n';
            *out << "d RUN_TIME " << response.wall_time() << std::endl;
            *out << "d MEMORY " << sysinfo::MemoryUsageProcess() << std::endl;
            *out << "d STATUS "<<response.status() << std::endl;
            *out << "d OBJECTIVE "<<response.objective_value() << std::endl;
            *out << "d BEST_BOUND "<<response.best_objective_bound() << std::endl;
            *out << "d BOOLEANS " << response.num_booleans() << std::endl;
            *out << "d CONFLICTS " << response.num_conflicts() << std::endl;
            *out << "d PROPAGATION " << response.num_binary_propagations() << std::endl;
            *out << "d INTEGER_PROPAGATION " << response.num_integer_propagations() << std::endl;
            *out << "d BRANCHES " << response.num_branches() << std::endl;
            if (response.status()== CpSolverStatus::OPTIMAL || response.status() == CpSolverStatus::FEASIBLE) {
                check(response, check_solution, strategy);
                if (eval)
                    eval_model();
            }
        }

        void eval_model(){
            double accuracy_train, accuracy_test, accuracy_train_bis, accuracy_test_bis;
            *out << " c starting evaluation..." << '\n';
            Evaluation test(weights_solution, bnn_data);
            *out << " c Testing accuracy with strong classification criterion : "<< '\n';
            accuracy_test = test.run_evaluation(true, true);
            *out << " c Training accuracy with strong classification criterion : "<< '\n';
            accuracy_train = test.run_evaluation(false, true);
            *out << " c Testing accuracy with weak classification criterion : "<< '\n';
            accuracy_test_bis = test.run_evaluation(true, false);
            *out << " c Training accuracy with weak classification criterion : "<< '\n';
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

            *out << "d TEST_STRONG_ACCURACY " << accuracy_test << std::endl;
            *out << "d TRAIN_STRONG_ACCURACY " << accuracy_train << std::endl;
            *out << "d TEST_WEAK_ACCURACY " << accuracy_test_bis << std::endl;
            *out << "d TRAIN_WEAK_ACCURACY " << accuracy_train_bis << std::endl;
        }

        void print_solution(const CpSolverResponse &r, const int &verbose, const int &index = 0){

            assert(index >=0);
            assert (verbose);
            if(r.status() == CpSolverStatus::OPTIMAL || r.status() == CpSolverStatus::FEASIBLE){

                int nb_layers = bnn_data.get_layers();
                if (verbose >1)
                {
                    *out << "\n s Solution "<< index << " : \n";

                    *out << "   Weights" << '\n';
                    for (size_t l = 1; l < nb_layers; ++l) {
                        *out << "   Layer "<< l << ": \n";
                        int size_previous_layer = bnn_data.get_archi(l-1);
                        for (size_t i = 0; i < size_previous_layer; ++i) {
                            int size_current_layer = bnn_data.get_archi(l);
                            for (size_t j = 0; j < size_current_layer; ++j) {
                                *out << "\t w["<<l<<"]["<<i<<"]["<<j<<"] = " << weights_solution[l-1][i][j];
                            }
                            *out << '\n';
                        }
                        *out << '\n';
                    }
                }
                for (size_t i = 0; i < nb_examples; i++) {
                    *out << " s Example "<< i ;
                    if (verbose >1)
                        *out << " \n" ;
                    if (verbose >1)
                        *out << "   Input : " << '\n';
                    if (verbose >1)
                        for (size_t j = 0; j < 784; j++) {
                            *out << (int)inputs[i][j] << " ";
                        }
                    if (verbose >1)
                        *out << " \n" ;
                    *out << "  Label : "<< labels[i] ;
                    if (verbose >1)
                        *out << " \n" ;
                    if (optimization_problem == '2')
                        *out << "   Classification : " << SolutionIntegerValue(r, classification[i]) << '\n';
                    else
                        *out << "   Classification : " << 1 << '\n';
                }

            }
            if(r.status()==CpSolverStatus::MODEL_INVALID){
                LOG(INFO) << ValidateCpModel(cp_model_builder.Build());
            }
        }
    };
  }
}

#endif
