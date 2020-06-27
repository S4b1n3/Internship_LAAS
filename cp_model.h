#ifndef EXAMPLES_CPP_CP_MODEL_H_
#define EXAMPLES_CPP_CP_MODEL_H_

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
#include <stdlib.h>

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>


namespace operations_research{
namespace sat{


class CP_Model {

	/* CPModel_MinWeight contains the constraint programming model for the full classification and minweight problem
      Atributs
      - bnn_data : data of the problem
      - cp_model_builder : the CP-SAT model proto
      - nb_examples : number of examples to test
      - weights : variable of the problem that represents the value of the weight
        for each arc between the neurons
      - activation : the value of the activation of each neuron on each layer (except hte first one)
      - preactivation : the value of the preactivation of each neuron on each layer
      - activation_first_layer : the value of the activation of each neuron on the first layer
      - domain : the domain of the variables [-1, 1]
      - activation_domain : the domain of the activation variables (-1,1)
      - objectif : LinearExpr used to find the objective
      - response : contains the informations of the solver after calling
      - model : used to print all the solutions
      - parameters : parameters for the sat solver
      - file_out : name of the output file
      - file : ostream used to manipulate the output file
      - index_rand : random number used to select the example
      - prod_constraint : boolean that indicates which constraints to use to compute the preactivation
      - output_path : path of the output files (results and solution)
	 */

protected:

	Data bnn_data;
	//Mohamed: Its is confusing to declare an object cp_model inside a class called Cp_model --> change this
	CpModelBuilder cp_model_builder;
	int nb_examples;

	std::vector<std::vector<uint8_t>> inputs;
	std::vector<int> labels;

	//weights[a][b][c] is the weight variable of the arc between neuron b on layer a-1 and neuron c on layer a
	std::vector<std::vector <std::vector<IntVar>>> weights;
	std::vector<std::vector <std::vector<int>>> weights_solution;

	std::vector <std::vector<std::vector<IntVar>>> activation;
	std::vector <std::vector<std::vector<IntVar>>> preactivation;
	std::vector <std::vector<int>> activation_solution;
	std::vector <std::vector<int>> preactivation_solution;
	//ORTools requires coefitions to be int64
	std::vector<std::vector<int64>> activation_first_layer;

	const Domain domain;
	const Domain activation_domain;
	LinearExpr objectif;
	CpSolverResponse response;
	Model model;
	SatParameters parameters;

	const std::string file_out;
	const std::string file_out_extension;
	std::ofstream file;

	int index_rand;
	const bool prod_constraint;
	const std::string output_path;


public:

	/*
        Constructor of the class CPModel
        Argument :
        - a vector representing the architecture of a BNN
        - nb_examples : number of examples to test
        - _prod_constraint : boolean that indicates which constraints to use to compute the preactivation
        The constructor initialize the data of the problem and the domain of the variables
        Call the constructor launch the method to solve the problem
	 */
	CP_Model(const std::vector<int> &_archi, const int &_nb_examples, const bool _prod_constraint, const std::string &_output_path):
		bnn_data(_archi), domain(-1,1), activation_domain(Domain::FromValues({-1,1})), file_out("tests/solver_solution_"),
		file_out_extension(".tex"), nb_examples(_nb_examples), prod_constraint(_prod_constraint), output_path(_output_path){
		std::cout << "number of layers : "<<bnn_data.get_layers() << '\n';
		bnn_data.print_archi();
		bnn_data.print_dataset();
		index_rand = rand()%(60000-_nb_examples);
		inputs.resize(nb_examples);
		labels.resize(nb_examples);
		for (size_t i = 0; i < nb_examples; i++) {
			inputs[i] = bnn_data.get_dataset().training_images[index_rand+i];
			labels[i] = (int)bnn_data.get_dataset().training_labels[index_rand+i];
		}
	}

	CP_Model(const int &_nb_examples_per_label, const std::vector<int> &_archi, const bool _prod_constraint, const std::string &_output_path):
		bnn_data(_archi), domain(-1,1), activation_domain(Domain::FromValues({-1,1})), file_out("tests/solver_solution_"),
		file_out_extension(".tex"), nb_examples(10*_nb_examples_per_label), prod_constraint(_prod_constraint), output_path(_output_path){
		std::cout << "number of layers : "<<bnn_data.get_layers() << '\n';
		bnn_data.print_archi();
		bnn_data.print_dataset();
		//index_rand = rand()%50000;

		std::clock_t c_start = std::clock();

		int compt_ex = 0, j = 0;
		std::vector<int> occ(10);
		std::vector<int> ind;
		while (compt_ex < nb_examples) {
			index_rand = rand()%60000;
			auto it = std::find(std::begin(ind), std::end(ind), index_rand);
			if (it == ind.end()) {
				ind.push_back(index_rand);
				int label = (int)bnn_data.get_dataset().training_labels[j];
				if(occ[label] < _nb_examples_per_label){
					inputs.push_back(bnn_data.get_dataset().training_images[j]);
					labels.push_back(label);
					compt_ex++;
					occ[label]++;
				}
			}
			j++;
		}

		std::clock_t c_end = std::clock();
		std::cout << " Building dataset finished; CPU setup time is " << (c_end-c_start) / CLOCKS_PER_SEC << " s" <<std::endl;
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

	std::vector<std::vector <std::vector<int>>> get_weights_solution() const{
		return weights_solution;
	}

	std::vector<std::vector <int>> get_preactivation_solution() const{
		return preactivation_solution;
	}

	std::vector<std::vector <int>> get_activation_solution() const{
		return activation_solution;
	}

	/* declare_activation_variable method
        Parameters :
        - index_example : index of the training example to classifie
        Output : None
        n_{lj} variables from the CP paper
	 */
	void declare_activation_variables(const int &index_example){
		assert(index_example>=0);
		assert(index_example<nb_examples);

		int size = inputs[index_example].size();
		activation_first_layer.resize(nb_examples);

		activation_first_layer[index_example].resize(size);
		for (size_t i = 0; i < size; i++) {
			activation_first_layer[index_example][i] = (int)inputs[index_example][i];
		}



		activation.resize(nb_examples);
		int tmp = bnn_data.get_layers()-1;
		activation[index_example].resize(tmp);
		for (size_t l = 0; l < tmp; ++l) {
			int tmp2 = bnn_data.get_archi(l+1);
			activation[index_example][l].resize(tmp2);
			for(size_t j = 0; j < tmp2; ++j){
				activation[index_example][l][j] = cp_model_builder.NewIntVar(activation_domain);
			}
		}
	}


	/* declare_preactivation_variable method
        Parameters :
        - index_example : index of the training example to classifie
        Output : None
        preactivation[l] represents the preactivation of layer l+1 where l \in [0,bnn_data.get_layers()-1]
	 */
	void declare_preactivation_variables(const int &index_example){
		assert(index_example>=0);
		assert(index_example<nb_examples);
		preactivation.resize(nb_examples);

		int sum_image = 0;
		for(std::vector<uint8_t>::iterator it = inputs[index_example].begin(); it != inputs[index_example].end(); ++it)
	    sum_image += (int)*it;

		int tmp = bnn_data.get_layers()-1;
		preactivation[index_example].resize(tmp);
		for (size_t l = 0; l < tmp; l++) {
			int tmp2 = bnn_data.get_archi(l+1);
			preactivation[index_example][l].resize(tmp2);
			int tmp3 = bnn_data.get_archi(l);
			for(size_t j = 0; j < tmp2; j++){
				if(l == 0){
					preactivation[index_example][l][j] = cp_model_builder.NewIntVar(Domain(-sum_image, sum_image));
				}
				else {
					preactivation[index_example][l][j] = cp_model_builder.NewIntVar(Domain(-tmp3, tmp3));
				}
			}
		}
	}

	/* get_a_lj method
        Parameters :
        - index_example : index of the example to classifie
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


	/* declare_weight_variables method
        This method initialize the weight variables
        weights[a][b][c] is the weight variable of the edge between neuron b on layer a-1 and neuron c on layer a
        Parameters : None
        Output : None
	 */
	void declare_weight_variables() {

		//Initialization of the variables

		int tmp = bnn_data.get_layers();
		weights.resize(tmp);
		for (size_t l = 1; l < tmp; l++) {
			int tmp2 = bnn_data.get_archi(l-1);
			weights[l-1].resize(tmp2);
			for(size_t i = 0; i < tmp2; i++){
				int tmp3 = bnn_data.get_archi(l);
				weights[l-1][i].resize(tmp3);
				for (size_t j = 0; j < tmp3; j++) {

					/*One weight for each connection between the neurons i of layer
                  l-1 and the neuron j of layer l : N(i) * N(i+1) connections*/

					weights[l-1][i][j] = cp_model_builder.NewIntVar(domain);
				}
			}
		}
	}


	/* get_w_ilj method
        Parameters :
        - i : neuron on layer l-1 \in [0, bnn_data.get_archi(l-1)]
        - l : layer \in [1, bnn_data.get_layers()]
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
		return weights[l-1][i][j];
	}

	virtual void model_declare_objective() = 0;

	/* model_activation_constraint method
        Parameters :
        - index_example : index of the example to classifie
        - l : layer \in [1, bnn_data.get_layers()]
        - j : neuron on layer l \in [0, bnn_data.get_archi(l)]

        preactivation[l][j] >= 0 => activation[l][j] = 1
        preactivation[l][j] < 0 => activation[l][j] = -1
        Output : None
	 */
	void model_activation_constraint(const int &index_example, const int &l, const int &j){
		/*assert (index_example>=0);
		assert (index_example<nb_examples);
		assert (l>0);
		assert (l<bnn_data.get_layers());
		assert (j>=0);
		assert (j<bnn_data.get_archi(l));*/

		//_temp_bool is true iff preactivation[l][j] < 0
		//_temp_bool is false iff preactivation[l][j] >= 0
		BoolVar _temp_bool = cp_model_builder.NewBoolVar();
		cp_model_builder.AddLessThan(get_a_lj(index_example, l, j), 0).OnlyEnforceIf(_temp_bool);
		cp_model_builder.AddGreaterOrEqual(get_a_lj(index_example, l, j), 0).OnlyEnforceIf(Not(_temp_bool));
		cp_model_builder.AddEquality(activation[index_example][l-1][j], -1).OnlyEnforceIf(_temp_bool);
		cp_model_builder.AddEquality(activation[index_example][l-1][j], 1).OnlyEnforceIf(Not(_temp_bool));
	}

	/* model_preactivation_constraint method
        Parameters :
        - index_example : index of the example to classifie
        - l : layer \in [1, bnn_data.get_layers()-1]
        - j : neuron on layer l \in [0, bnn_data.get_archi(l)]
        Output : None
	 */
	void model_preactivation_constraint(const int &index_example, const int &l, const int &j){
		//assert(index_example>=0);
		//assert(index_example<nb_examples);
		//No need for this
		//assert(l>0);
		//No need for this
		//assert(l<bnn_data.get_layers());
		//No need for this
		//assert(j>=0);
		//No need for this
		//assert(j<bnn_data.get_archi(l));
		if(l == 1){
			LinearExpr temp(0);
			int tmp = bnn_data.get_archi(0);
			for (size_t i = 0; i < tmp; i++) {
				temp.AddTerm(get_w_ilj(i, l, j), activation_first_layer[index_example][i]);
			}
			cp_model_builder.AddEquality(get_a_lj(index_example, 1, j), temp);
		}
		else{
			std::vector<IntVar> temp(bnn_data.get_archi(l-1));
			int tmp = bnn_data.get_archi(l-1) ;
			for (size_t i = 0; i < tmp; i++) {
				temp[i] = cp_model_builder.NewIntVar(domain);
				if(!prod_constraint){

					IntVar sum_weights_activation = cp_model_builder.NewIntVar(Domain(-2,2));
					IntVar sum_temp_1 = cp_model_builder.NewIntVar(Domain(0, 2));
					cp_model_builder.AddEquality(sum_weights_activation, LinearExpr::Sum({get_w_ilj(i, l, j), activation[index_example][l-2][i]}));
					cp_model_builder.AddEquality(sum_temp_1, temp[i].AddConstant(1));
					cp_model_builder.AddAbsEquality(sum_temp_1, sum_weights_activation);

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

					BoolVar b1 = cp_model_builder.NewBoolVar();
					BoolVar b2 = cp_model_builder.NewBoolVar();

					// Implement b1 == (temp[i] == 0)
					cp_model_builder.AddEquality(temp[i], 0).OnlyEnforceIf(b1);
					cp_model_builder.AddNotEqual(temp[i], LinearExpr(0)).OnlyEnforceIf(Not(b1));
					//Implement b2 == (weights == 0)
					cp_model_builder.AddEquality(get_w_ilj(i, l, j), 0).OnlyEnforceIf(b2);
					cp_model_builder.AddNotEqual(get_w_ilj(i, l, j), LinearExpr(0)).OnlyEnforceIf(Not(b2));

					// b1 implies b2 and b2 implies b1
					cp_model_builder.AddImplication(b2, b1);
					cp_model_builder.AddImplication(b1, b2);

					BoolVar b3 = cp_model_builder.NewBoolVar();
					BoolVar b4 = cp_model_builder.NewBoolVar();

					// Implement b3 == (temp[i] == 1)
					cp_model_builder.AddEquality(temp[i], 1).OnlyEnforceIf(b3);
					cp_model_builder.AddNotEqual(temp[i], LinearExpr(1)).OnlyEnforceIf(Not(b3));
					//Implement b4 == (weights == activation)
					cp_model_builder.AddEquality(get_w_ilj(i, l, j), activation[index_example][l-2][i]).OnlyEnforceIf(b4);
					cp_model_builder.AddNotEqual(get_w_ilj(i, l, j), activation[index_example][l-2][i]).OnlyEnforceIf(Not(b4));


					// b3 implies b4 and b4 implies b3
					cp_model_builder.AddImplication(b3, b4);
					cp_model_builder.AddImplication(b4, b3);

				}
			}
			cp_model_builder.AddEquality(get_a_lj(index_example, l, j), LinearExpr::Sum(temp));
		}
	}


	virtual void model_output_constraint(const int &index_examples) = 0;



	void setup_branching(std::string strategy){


		std::vector<IntVar> w;
		int tmp = bnn_data.get_layers();
		for (size_t l = 1; l < tmp; l++){
			int tmp2 = bnn_data.get_archi(l-1);
			for(size_t i = 0; i < tmp2; i++){
				int tmp3 = bnn_data.get_archi(l);
				for (size_t j = 0; j < tmp3; j++)
					w.push_back( weights[l-1][i][j] );
			}
		}

		std::vector<IntVar> reverse_w(w);
		std::reverse(std::begin(reverse_w), std::end(reverse_w));



		std::cout << " c number of branching variables is "  << w.size()  << std::endl;

		if (strategy == "lex_max_0"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MAX_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  PORTFOLIO_WITH_QUICK_RESTART_SEARCH  SELECT_MAX_VALUE "  << std::endl;
			//parameters.set_search_branching(SatParameters::PORTFOLIO_WITH_QUICK_RESTART_SEARCH );
		}


		if (strategy == "lex_max_1"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MAX_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  PORTFOLIO_WITH_QUICK_RESTART_SEARCH  SELECT_MAX_VALUE "  << std::endl;
			parameters.set_search_branching(SatParameters::PORTFOLIO_WITH_QUICK_RESTART_SEARCH );
		}

		if (strategy == "lex_max_2"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MAX_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  AUTOMATIC_SEARCH  SELECT_MAX_VALUE "  << std::endl;
			parameters.set_search_branching(SatParameters::AUTOMATIC_SEARCH );
		}
		if (strategy == "lex_max_3"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MAX_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  HINT_SEARCH SELECT_MAX_VALUE"  << std::endl;
			parameters.set_search_branching(SatParameters::HINT_SEARCH );
		}

		if (strategy == "lex_max_4"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MAX_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  FIXED_SEARCH SELECT_MAX_VALUE"  << std::endl;
			parameters.set_search_branching(SatParameters::FIXED_SEARCH );
		}

		if (strategy == "antilex_max_0"){
			cp_model_builder.AddDecisionStrategy(reverse_w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MAX_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching on reverse_w : lex  PORTFOLIO_WITH_QUICK_RESTART_SEARCH  SELECT_MAX_VALUE "  << std::endl;
			//parameters.set_search_branching(SatParameters::PORTFOLIO_WITH_QUICK_RESTART_SEARCH );
		}


		if (strategy == "antilex_max_1"){
			cp_model_builder.AddDecisionStrategy(reverse_w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MAX_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching on reverse_w : lex  PORTFOLIO_WITH_QUICK_RESTART_SEARCH  SELECT_MAX_VALUE "  << std::endl;
			parameters.set_search_branching(SatParameters::PORTFOLIO_WITH_QUICK_RESTART_SEARCH );
		}

		if (strategy == "antilex_max_2"){
			cp_model_builder.AddDecisionStrategy(reverse_w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MAX_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching on reverse : lex  AUTOMATIC_SEARCH  SELECT_MAX_VALUE "  << std::endl;
			parameters.set_search_branching(SatParameters::AUTOMATIC_SEARCH );
		}
		if (strategy == "antilex_max_3"){
			cp_model_builder.AddDecisionStrategy(reverse_w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MAX_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching on reverse w: lex  HINT_SEARCH SELECT_MAX_VALUE"  << std::endl;
			parameters.set_search_branching(SatParameters::HINT_SEARCH );
		}

		if (strategy == "antilex_max_4"){
			cp_model_builder.AddDecisionStrategy(reverse_w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MAX_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching on reverse_w : lex  FIXED_SEARCH SELECT_MAX_VALUE"  << std::endl;
			parameters.set_search_branching(SatParameters::FIXED_SEARCH );
		}




		if (strategy == "lex_median_0"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MEDIAN_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  PORTFOLIO_WITH_QUICK_RESTART_SEARCH  SELECT_MEDIAN_VALUE "  << std::endl;
			//parameters.set_search_branching(SatParameters::PORTFOLIO_WITH_QUICK_RESTART_SEARCH );
		}


		if (strategy == "lex_median_1"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MEDIAN_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  PORTFOLIO_WITH_QUICK_RESTART_SEARCH  SELECT_MEDIAN_VALUE "  << std::endl;
			parameters.set_search_branching(SatParameters::PORTFOLIO_WITH_QUICK_RESTART_SEARCH );
		}

		if (strategy == "lex_median_2"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MEDIAN_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  AUTOMATIC_SEARCH  SELECT_MEDIAN_VALUE "  << std::endl;
			parameters.set_search_branching(SatParameters::AUTOMATIC_SEARCH );
		}
		if (strategy == "lex_median_3"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MEDIAN_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  HINT_SEARCH SELECT_MEDIAN_VALUE"  << std::endl;
			parameters.set_search_branching(SatParameters::HINT_SEARCH );
		}

		if (strategy == "lex_median_4"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MEDIAN_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  FIXED_SEARCH SELECT_MEDIAN_VALUE"  << std::endl;
			parameters.set_search_branching(SatParameters::FIXED_SEARCH );
		}

		if (strategy == "antilex_median_0"){
			cp_model_builder.AddDecisionStrategy(reverse_w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MEDIAN_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching on reverse_w : lex  PORTFOLIO_WITH_QUICK_RESTART_SEARCH  SELECT_MEDIAN_VALUE "  << std::endl;
			//parameters.set_search_branching(SatParameters::PORTFOLIO_WITH_QUICK_RESTART_SEARCH );
		}


		if (strategy == "antilex_median_1"){
			cp_model_builder.AddDecisionStrategy(reverse_w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MEDIAN_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching on reverse_w : lex  PORTFOLIO_WITH_QUICK_RESTART_SEARCH  SELECT_MEDIAN_VALUE "  << std::endl;
			parameters.set_search_branching(SatParameters::PORTFOLIO_WITH_QUICK_RESTART_SEARCH );
		}

		if (strategy == "antilex_median_2"){
			cp_model_builder.AddDecisionStrategy(reverse_w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MEDIAN_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching on reverse : lex  AUTOMATIC_SEARCH  SELECT_MEDIAN_VALUE "  << std::endl;
			parameters.set_search_branching(SatParameters::AUTOMATIC_SEARCH );
		}
		if (strategy == "antilex_median_3"){
			cp_model_builder.AddDecisionStrategy(reverse_w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MEDIAN_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching on reverse w: lex  HINT_SEARCH SELECT_MEDIAN_VALUE"  << std::endl;
			parameters.set_search_branching(SatParameters::HINT_SEARCH );
		}

		if (strategy == "antilex_median_4"){
			cp_model_builder.AddDecisionStrategy(reverse_w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_MEDIAN_VALUE);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching on reverse_w : lex  FIXED_SEARCH SELECT_MEDIAN_VALUE"  << std::endl;
			parameters.set_search_branching(SatParameters::FIXED_SEARCH );
		}



		if (strategy == "lex5"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_LOWER_HALF);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  PORTFOLIO_WITH_QUICK_RESTART_SEARCH  SELECT_LOWER_HALF'"  << std::endl;
			parameters.set_search_branching(SatParameters::PORTFOLIO_WITH_QUICK_RESTART_SEARCH );
		}

		if (strategy == "lex6"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_LOWER_HALF);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  AUTOMATIC_SEARCH  SELECT_LOWER_HALF "  << std::endl;
			parameters.set_search_branching(SatParameters::AUTOMATIC_SEARCH );
		}
		if (strategy == "lex7"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_LOWER_HALF);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  HINT_SEARCH  SELECT_LOWER_HALF"  << std::endl;
			parameters.set_search_branching(SatParameters::HINT_SEARCH );
		}

		if (strategy == "lex8"){
			cp_model_builder.AddDecisionStrategy(w, DecisionStrategyProto::CHOOSE_FIRST,	DecisionStrategyProto::SELECT_LOWER_HALF);
			//std::cout << " setting branching :  FIXED_SEARCH  '"  << std::endl;
			std::cout << " setting branching : lex  FIXED_SEARCH  SELECT_LOWER_HALF"  << std::endl;
			parameters.set_search_branching(SatParameters::FIXED_SEARCH );
		}



	}



	/* run method
        This function calls all the necessary methods to run the solver
        Parameters :
        - nb_seconds : Sets a time limit of nb_seconds
        Output : None
	 */
	void run(const double &nb_seconds, std::string _strategy){
		std::cout<< " c declare variables and constraints " <<std::endl;

		std::clock_t c_start = std::clock();


		assert(nb_seconds>0);
		declare_weight_variables();                      //initialization of the variables
		for (size_t i = 0; i < nb_examples; i++) {
			declare_preactivation_variables(i);
			declare_activation_variables(i);
		}
		for (size_t i = 0; i < nb_examples; i++) {
			int tmp = bnn_data.get_layers();
			for (size_t l = 1; l < tmp; l++) {
				int tmp2 =  bnn_data.get_archi(l);
				for (size_t j = 0; j < tmp2; j++) {
					model_preactivation_constraint(i, l, j);
					model_activation_constraint(i, l, j);
				}
			}
		}
		for (size_t i = 0; i < nb_examples; i++) {
			model_output_constraint(i);
		}
		model_declare_objective() ;                 //initialization of the objective
		setup_branching(_strategy) ;
		parameters.set_max_time_in_seconds(nb_seconds);     //Add a timelimit
		parameters.set_random_seed(1000);
		model.Add(NewSatParameters(parameters));                       //objective function
		// your_algorithm
		std::clock_t c_end = std::clock();

		//long_double time_elapsed_ms = 1000.0 * ;

		std::string result_file = output_path+"/results"+std::to_string(nb_examples)+".stat";
		std::ofstream parser(result_file.c_str(), std::ios::app);
		parser << "setup time " << (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
		parser.close();

		std::cout << " c Setup finished; CPU setup time is " << (c_end-c_start) / CLOCKS_PER_SEC << " s" <<std::endl;
		std::cout<< " c running the solver.. " <<std::endl;
	}

	void check(const CpSolverResponse &r, const bool &check_solution, const int &index=0){

		int tmp = bnn_data.get_layers();
		weights_solution.resize(tmp);
		for (size_t l = 1; l < tmp; ++l) {
			int tmp2 = bnn_data.get_archi(l-1);
			weights_solution[l-1].resize(tmp2);
			for (size_t i = 0; i < tmp2; ++i) {
				int tmp3 = bnn_data.get_archi(l);
				weights_solution[l-1][i].resize(tmp3);
				for (size_t j = 0; j < tmp3; ++j) {
					weights_solution[l-1][i][j] = SolutionIntegerValue(r, weights[l-1][i][j]);
				}
			}
		}


		for (size_t i = 0; i < nb_examples; i++) {

			preactivation_solution.resize(tmp-1);
			for (size_t l = 0; l < tmp-1; l++) {
				int tmp2 = bnn_data.get_archi(l+1);
				preactivation_solution[l].resize(tmp2);
				for(size_t j = 0; j < tmp2; j++){
					preactivation_solution[l][j] = SolutionIntegerValue(r, preactivation[i][l][j]);
				}
			}

			activation_solution.resize(tmp);
			for (size_t l = 0; l < tmp; l++) {
				int tmp2 = bnn_data.get_archi(l);
				activation_solution[l].resize(tmp2);
				for(size_t j = 0; j < tmp2; j++){
					if(l == 0){
						activation_solution[l][j] = (int)activation_first_layer[i][j];
					}
					else{
						activation_solution[l][j] = SolutionIntegerValue(r, activation[i][l-1][j]);
					}
				}
			}
			if (check_solution) {
				Solution check_solution(bnn_data, weights_solution, activation_solution, preactivation_solution, i+index_rand);
				std::cout << "Checking solution : "<<index<<" : ";
				bool checking = check_solution.run_solution(true);
			}
		}
	}


	/*print_header_solution method
        This function writes on the output file the latex header
        Parameters :
        - num_sol : the index of the solution
        Output ; None
	 */
	void print_header_solution(const int &num_sol){
		assert(num_sol>=0);
		file.open(file_out+std::to_string(num_sol)+file_out_extension, std::ios::out);
		if (file.bad()) std::cout<<"Error oppening file"<<std::endl;
		else{
			file <<"\\documentclass{article}"<<std::endl;
			file <<"\\usepackage{tikz}"<<std::endl;
			file <<"\\usetikzlibrary{arrows.meta}"<<std::endl;
			file <<"\\begin{document}"<<std::endl;
			file <<"\\begin{tikzpicture}"<<std::endl;
		}
		file.close();
	}

	/* print_node mehod
        This function creates a node in latex
        Parameters :
        - name : name of the node
        - x, y : position of the node
        Output : a string containing the latex command that will create the node
	 */
	std::string print_node(const std::string &name, const int &x, const int &y){
		assert(x >= 0);
		assert(y >= 0);
		return "\\node ("+name+") at ("+std::to_string(x)+","+std::to_string(y)+") {"+name+"};";
	}

	/* print_arc mehod
        This function creates an arc in latex
        Parameters :
        - origin : origin node of the arc
        - target : target node of the arc
        - weight : value of the weight for this arc
        Output : a string containing the latex command that will create the arc
	 */
	std::string print_arc(const std::string &origin, const std::string &target, const int &weight){
		assert(weight>=-1);
		assert(weight<=1);
		return "\\path [->] ("+origin+") edge node {$"+std::to_string(weight)+"$} ("+target+");";
	}

	// Print some statistics from the solver: Runtime, number of nodes, number of propagation (filtering, pruning), memory,
	// Status: Optimal, suboptimal, satisfiable, unsatisfiable, unkown
	// Output Status: {OPTIMAL, FEASIBLE, INFEASIBLE, MODEL_INVALID, UNKNOWN}
	int print_statistics(const int &check_solution){
		response = SolveCpModel(cp_model_builder.Build(), &model);
		std::string result_file = output_path+"/results"+std::to_string(nb_examples)+".stat";
		std::ofstream parser(result_file.c_str(), std::ios::app);
		std::cout << "\nSome statistics on the solver response : " << '\n';
		LOG(INFO) << CpSolverResponseStats(response);
		std::cout << "\nSome statistics on the model : " << '\n';
		LOG(INFO) << CpModelStats(cp_model_builder.Build());
		if(parser){
			parser << std::endl << "run time " << response.wall_time() << std::endl;
			parser << "memory " << sysinfo::MemoryUsageProcess() << std::endl;
			std::cout << "Memory : "<< sysinfo::MemoryUsageProcess() << '\n';
			parser << "status "<<response.status() << std::endl;
			if (response.status()== CpSolverStatus::OPTIMAL)
				parser << "objective "<<response.objective_value() << std::endl;
			else
				parser << "objective "<<response.objective_value() << std::endl;
			parser << "best bound "<<response.best_objective_bound() << std::endl;
			parser << "booleans " << response.num_booleans() << std::endl;
			parser << "conflicts " << response.num_conflicts() << std::endl;
			parser << "propagation " << response.num_binary_propagations() << std::endl;
			parser << "integer propagation " << response.num_integer_propagations() << std::endl;
			parser << "branches " << response.num_branches() << std::endl;
			parser << CpModelStats(cp_model_builder.Build()) << std::endl;
			parser << std::endl;
			parser.close();
		}
		else
			std::cout << "Error opening parser file" << '\n';
		if (response.status()== CpSolverStatus::OPTIMAL || response.status() == CpSolverStatus::FEASIBLE) {
			check(response, check_solution);
		}
		return response.status();
	}



	virtual void print_solution(const CpSolverResponse &r, const int &index = 0) = 0;

	/* print_solution method
        This function prints a solution returned by the solver
        if this solution is feasible or optimal
        Parameters :
        - r : response of the solver
        - index : index of the solution (default : 0)
        Output : None
	 */
	void print_solution_bis(const CpSolverResponse &r, const int &index = 0){
		assert(index >= 0);
		if(r.status() == CpSolverStatus::OPTIMAL || r.status() == CpSolverStatus::FEASIBLE){
			print_header_solution(index);
			file.open(file_out+std::to_string(index)+file_out_extension, std::ios::app);
			if (file.bad()) std::cout<<"Erreur ouverture"<<std::endl;
			else{
				file <<"\\begin{scope}[every node/.style={circle,thick,draw}]" << std::endl;
				int height = 0;
				for (size_t l = 0; l < bnn_data.get_layers(); l++) {
					for (size_t i = 0; i < bnn_data.get_archi(l); i++) {
						std::string name("N"+std::to_string(l)+std::to_string(i));
						file << print_node(name, 2*l, height)<<std::endl;
						height -= 2;
					}
					height = 0;
				}
				file << "\\end{scope}"<<std::endl;
				file << "\\begin{scope}[>={Stealth[black]}, every node/.style={fill=white,circle}, every edge/.style={draw=red,very thick}]" << std::endl;
				for (size_t l = 1; l < bnn_data.get_layers(); l++) {
					for (size_t i = 0; i < bnn_data.get_archi(l-1); i++) {
						for (size_t j = 0; j < bnn_data.get_archi(l); j++) {
							std::string origin("N"+std::to_string(l-1)+std::to_string(i));
							std::string target("N"+std::to_string(l)+std::to_string(j));
							file << print_arc(origin, target, SolutionIntegerValue(r, weights[l-1][i][j]))<<std::endl;
						}
					}
				}

				file << "\\end{scope}"<<std::endl;
				file <<"\\end{tikzpicture}"<<std::endl;
				file <<"\\end{document}"<<std::endl;
			}
			file.close();


		}
		if(r.status()==CpSolverStatus::MODEL_INVALID){
			LOG(INFO) << ValidateCpModel(cp_model_builder.Build());
		}
	}

	/* print_all_solutions method
        This function prints all feasible or optimal solutions returned by the solver
        Parameters : None
        Output : None
	 */
	void print_all_solutions(){
		int num_solutions = 0;
		Model _model;
		_model.Add(NewFeasibleSolutionObserver([&](const CpSolverResponse& r) {
			print_solution_bis(r, num_solutions);
			num_solutions++;
		}));
		parameters.set_enumerate_all_solutions(true);
		_model.Add(NewSatParameters(parameters));
		response = SolveCpModel(cp_model_builder.Build(), &_model);
		LOG(INFO) << "Number of solutions found: " << num_solutions;
	}

} ; //close class CPModel
} //close namespace sat
} //close namespace operations_research



#endif /* EXAMPLES_CPP_cp_model_builder_H_ */
