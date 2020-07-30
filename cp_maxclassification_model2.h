#ifndef EXAMPLES_CPP_CP_MAXCLASSIFICATION2_MODEL_H_
#define EXAMPLES_CPP_CP_MAXCLASSIFICATION2_MODEL_H_

#include "ortools/sat/cp_model.h"
#include "ortools/sat/model.h"
#include "ortools/sat/sat_parameters.pb.h"

#include "ortools/util/sorted_interval_list.h"
#include "ortools/sat/cp_model_checker.h"

#include "cp_model.h"


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

    /* CPModel_MaxClassification contains the constraint programming model for the max classification problem
    Atributs
    - bnn_data : data of the problem
    - cp_model : the CP-SAT model proto
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

    class CPModel_MaxClassification2 : public CP_Model{
      protected :
        std::vector<BoolVar> classification;
        std::vector<int> classification_solution;

      public :

      /*
      Constructor of the class CPModel
      Argument :
      - a vector representing the architecture of a BNN
      - nb_examples : number of examples to test
      - _prod_constraint : boolean that indicates which constraints to use to compute the preactivation
      The constructor initialize the data of the problem and the domain of the variables
      Call the constructor launch the method to solve the problem
      */
      CPModel_MaxClassification2(Data *_data, const int &_nb_examples, const bool _prod_constraint, const std::string &_output_path):
                        CP_Model(_data, _nb_examples, _prod_constraint, _output_path){

      }

      CPModel_MaxClassification2(const int &_nb_examples_per_label, Data *_data, const bool _prod_constraint, const std::string &_output_path):
                        CP_Model(_nb_examples_per_label, _data, _prod_constraint, _output_path){

      }

      CPModel_MaxClassification2(Data *_data, const bool _prod_constraint, const std::string &_output_path, std::vector<std::vector<std::vector<int>>> _weights, const std::vector<int> &_indexes_examples):
                      CP_Model(_data, _prod_constraint, _output_path, _weights, _indexes_examples){

      }

      CPModel_MaxClassification2(Data *_data, const bool _prod_constraint, const std::string &_output_path, const std::string &_input_file):
                      CP_Model(_data, _prod_constraint, _output_path, _input_file){

      }

      CPModel_MaxClassification2(Data *_data, const bool _prod_constraint, const std::string &_output_path, const std::string &_input_file, const std::string &_solution_file):
                      CP_Model(_data, _prod_constraint, _output_path, _input_file, _solution_file){

      }

      void model_activation_constraint(const int &index_example, const int &l, const int &j){
    		/*assert (index_example>=0);
    		assert (index_example<nb_examples);
    		assert (l>0);
    		assert (l<bnn_data->get_layers());
    		assert (j>=0);
    		assert (j<bnn_data->get_archi(l));*/

    		//_temp_bool is true iff preactivation[l][j] < 0
    		//_temp_bool is false iff preactivation[l][j] >= 0
    		BoolVar _temp_bool = cp_model_builder.NewBoolVar();
    		cp_model_builder.AddLessThan(get_a_lj(index_example, l, j), 0).OnlyEnforceIf({_temp_bool, classification[index_example]});
    		cp_model_builder.AddGreaterOrEqual(get_a_lj(index_example, l, j), 0).OnlyEnforceIf({Not(_temp_bool), classification[index_example]});
    		cp_model_builder.AddEquality(activation[index_example][l-1][j], -1).OnlyEnforceIf({_temp_bool, classification[index_example]});
    		cp_model_builder.AddEquality(activation[index_example][l-1][j], 1).OnlyEnforceIf({Not(_temp_bool), classification[index_example]});
    	}

    	/* model_preactivation_constraint method
            Parameters :
            - index_example : index of the example to classifie
            - l : layer \in [1, bnn_data->get_layers()-1]
            - j : neuron on layer l \in [0, bnn_data->get_archi(l)]
            Output : None
    	 */
    	void model_preactivation_constraint(const int &index_example, const int &l, const int &j){
    		//assert(index_example>=0);
    		//assert(index_example<nb_examples);
    		//No need for this
        assert(l>0);
    		//No need for this
    		assert(l<bnn_data->get_layers());
    		//No need for this
    		assert(j>=0);
    		//No need for this
    		assert(j<bnn_data->get_archi(l));

    		if(l == 1){
    			LinearExpr temp(0);
    			int tmp = bnn_data->get_archi(0);
    			for (size_t i = 0; i < tmp; i++) {
            if (activation_first_layer[index_example][i] != 0) {
    					temp.AddTerm(get_w_ilj(i, l, j), activation_first_layer[index_example][i]);
    				}
    			}
    			cp_model_builder.AddEquality(get_a_lj(index_example, 1, j), temp).OnlyEnforceIf(classification[index_example]);
    		}
    		else{
    			std::vector<IntVar> temp(bnn_data->get_archi(l-1));
    			int tmp = bnn_data->get_archi(l-1) ;
    			for (size_t i = 0; i < tmp; i++) {
    				temp[i] = cp_model_builder.NewIntVar(domain);
    				if(!prod_constraint){

    					IntVar sum_weights_activation = cp_model_builder.NewIntVar(Domain(-2,2));
    					IntVar sum_temp_1 = cp_model_builder.NewIntVar(Domain(0, 2));
    					cp_model_builder.AddEquality(sum_weights_activation, LinearExpr::Sum({get_w_ilj(i, l, j), activation[index_example][l-2][i]})).OnlyEnforceIf(classification[index_example]);
    					cp_model_builder.AddEquality(sum_temp_1, temp[i].AddConstant(1)).OnlyEnforceIf(classification[index_example]);
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

              cp_model_builder.AddEquality(temp[i], 0).OnlyEnforceIf({get_weight_is_0_ilj(i,l,j), classification[index_example]});
     					cp_model_builder.AddNotEqual(temp[i], 0).OnlyEnforceIf({Not(get_weight_is_0_ilj(i,l,j)), classification[index_example]});


    					BoolVar b3 = cp_model_builder.NewBoolVar();

    					// Implement b3 == (temp[i] == 1)
    					cp_model_builder.AddEquality(temp[i], 1).OnlyEnforceIf({b3, classification[index_example]});
    					cp_model_builder.AddNotEqual(temp[i], 1).OnlyEnforceIf({Not(b3), classification[index_example]});
    					//Implement b3 == (weights == activation)
    					cp_model_builder.AddEquality(get_w_ilj(i, l, j), activation[index_example][l-2][i]).OnlyEnforceIf({b3, classification[index_example]});
    					cp_model_builder.AddNotEqual(get_w_ilj(i, l, j), activation[index_example][l-2][i]).OnlyEnforceIf({Not(b3), classification[index_example]});

    				}
    			}
    			cp_model_builder.AddEquality(get_a_lj(index_example, l, j), LinearExpr::Sum(temp)).OnlyEnforceIf(classification[index_example]);
    		}
    	}


      void declare_classification_variable(){
        for (size_t i = 0; i < nb_examples; i++) {
          classification[i] = cp_model_builder.NewBoolVar();
        }
      }

      /* model_activation_constraint method
            Parameters :
            - index_example : index of the example to classifie
            - l : layer \in [1, bnn_data->get_layers()]
            - j : neuron on layer l \in [0, bnn_data->get_archi(l)]

            preactivation[l][j] >= 0 => activation[l][j] = 1
            preactivation[l][j] < 0 => activation[l][j] = -1
            Output : None
    	 */




      /* model_output_constraint method

         the example is well classified => activation[index_examples][bnn_data->get_layers()-2][label] == 1 and
                                           activation[index_examples][bnn_data->get_layers()-2][i] == -1

        classification[index_examples] = 1 => activation[index_examples][bnn_data->get_layers()-2][label] ==  1
        classification[index_examples] = 1 => last_layer == -9

      Parameters :
       - index_example : index of the example to classifie
      Output : None
      */
      void model_output_constraint(const int &index_examples){
        /*assert(index_examples >= 0);
        assert(index_example < nb_examples);*/

        LinearExpr last_layer(0);
        const int label = labels[index_examples];
        int tmp = bnn_data->get_archi(bnn_data->get_layers()-1);
        int tmp2 = bnn_data->get_layers()-2;
        for (size_t i = 0; i < tmp; i++) {
          if (i != label) {
            last_layer.AddVar(activation[index_examples][tmp2][i]);
          }
        }

        cp_model_builder.AddEquality(activation[index_examples][tmp2][label], 1).OnlyEnforceIf(classification[index_examples]);
        cp_model_builder.AddEquality(last_layer, -(tmp - 1)).OnlyEnforceIf(classification[index_examples]);
      }


      /* model_objective_maximize_classification method
      This function sums all the values of classification in the LinearExpr objectif
      Parameters : None
      Output : None
      */
      void model_declare_objective(){
        for (size_t i = 0; i < nb_examples; i++) {
          objectif.AddVar(classification[i]);
        }
      }

      /* run method
      This function calls all the necessary methods to run the solver
      Parameters :
      - nb_seconds : Sets a time limit of nb_seconds
      Output : None
      */
      void run(const double &nb_seconds , std::string _strategy){
        classification.resize(nb_examples);
        classification_solution.resize(nb_examples);
        declare_classification_variable();
        CP_Model::run(nb_seconds, _strategy);
        cp_model_builder.Maximize(objectif);                        //objective function
      }

      void check(const CpSolverResponse &r, const bool &check_solution, const std::string &strategy, const int &index=0){

    		int tmp = bnn_data->get_layers();
    		weights_solution.resize(tmp);
    		for (size_t l = 1; l < tmp; ++l) {
    			int tmp2 = bnn_data->get_archi(l-1);
    			weights_solution[l-1].resize(tmp2);
    			for (size_t i = 0; i < tmp2; ++i) {
    				int tmp3 = bnn_data->get_archi(l);
    				weights_solution[l-1][i].resize(tmp3);
    				for (size_t j = 0; j < tmp3; ++j) {
    					weights_solution[l-1][i][j] = SolutionIntegerValue(r, weights[l-1][i][j]);
    				}
    			}
    		}

        int check_count = 0;
    		for (size_t i = 0; i < nb_examples; i++) {

          classification_solution[i] = SolutionIntegerValue(r, classification[i]);

    			preactivation_solution.resize(tmp-1);
    			for (size_t l = 0; l < tmp-1; l++) {
    				int tmp2 = bnn_data->get_archi(l+1);
    				preactivation_solution[l].resize(tmp2);
    				for(size_t j = 0; j < tmp2; j++){
    					preactivation_solution[l][j] = SolutionIntegerValue(r, preactivation[i][l][j]);
    				}
    			}

    			activation_solution.resize(tmp);
    			for (size_t l = 0; l < tmp; l++) {
    				int tmp2 = bnn_data->get_archi(l);
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
          if (check_solution && classification_solution[i] == 1) {
            std::clock_t c_start = std::clock();
            Solution check_solution(bnn_data, weights_solution, activation_solution, preactivation_solution, labels[i], inputs[i]);
            std::clock_t c_end = std::clock();
            bool checking = check_solution.run_solution(true, false, false);
            if (!check_model) {
    					std::cout << " c Build Solution time " << (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
    					std::cout << " d CHECKING "<<i<<" : " << checking << std::endl;
    				}
            if (checking) {
    					check_count++;
    				}
    				std::ofstream parser(output_path.c_str(), std::ios::app);
    				parser << "d CHECKING "<<checking<<std::endl;
    				parser.close();
    			}
    		}
    		if (check_model && check_count == nb_examples) {
    			std::cout << " c VERIFICATION 1" << '\n';
    		}
    		if (check_model && check_count != nb_examples) {
    			std::cout << " c VERIFICATION 0" << '\n';
    		}
    	}


      void print_solution(const CpSolverResponse &r, const int &index = 0){
        assert(index >=0);
        if(r.status() == CpSolverStatus::OPTIMAL || r.status() == CpSolverStatus::FEASIBLE){
          int tmp = bnn_data->get_layers();
          std::cout << "\n s Solution "<< index << " : \n";
          std::cout << "   Weights" << '\n';
          for (size_t l = 1; l < tmp; ++l) {
            std::cout << "   Layer "<< l << ": \n";
            int tmp2 = bnn_data->get_archi(l-1);
            for (size_t i = 0; i < tmp2; ++i) {
              int tmp3 = bnn_data->get_archi(l);
              for (size_t j = 0; j < tmp3; ++j) {
                std::cout << "\t w["<<l<<"]["<<i<<"]["<<j<<"] = " << weights_solution[l-1][i][j];
              }
              std::cout << '\n';
            }
            std::cout << '\n';
          }
          for (size_t i = 0; i < nb_examples; i++) {
            std::cout << " s Example "<< i << '\n';
            std::cout << "   Input : " << '\n';
            for (size_t j = 0; j < 784; j++) {
              std::cout << (int)inputs[i][j] << " ";
            }
            std::cout << "\n   Label : "<< labels[i] << '\n';
            std::cout << "   Classification : " << SolutionIntegerValue(r, classification[i]) << '\n';
          }

        }
        if(r.status()==CpSolverStatus::MODEL_INVALID){
          LOG(INFO) << ValidateCpModel(cp_model_builder.Build());
        }
      }

    }; // close class CPModel

  } // close namespace sat
} // close namespace operations_research



#endif /* EXAMPLES_CPP_CP_MAXCLASSIFICATION2_MODEL_H_ */
