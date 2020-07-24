#ifndef EXAMPLES_CPP_CP_MAXSUM_WEIGHTS_EXAMPLE_MODEL_H_
#define EXAMPLES_CPP_CP_MAXSUM_WEIGHTS_EXAMPLE_MODEL_H_

#include "ortools/sat/cp_model.h"
#include "ortools/sat/model.h"
#include "ortools/sat/sat_parameters.pb.h"

#include "ortools/util/sorted_interval_list.h"
#include "ortools/sat/cp_model_checker.h"

#include "cp_model.h"
#include "cp_maxclassification_model.h"


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
    class CPModel_MaxSum : public CPModel_MaxClassification {

      private :
        std::vector<IntVar> sum_weights_example;
        std::vector<IntVar> temp_objective;

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
      CPModel_MaxSum(Data *_data, const int &_nb_examples, const bool _prod_constraint, const std::string &_output_path):
                        CPModel_MaxClassification(_data, _nb_examples, _prod_constraint, _output_path){
      }

      CPModel_MaxSum(const int &_nb_examples_per_label, Data *_data, const bool _prod_constraint, const std::string &_output_path):
                        CPModel_MaxClassification(_nb_examples_per_label, _data, _prod_constraint, _output_path){

      }

      CPModel_MaxSum(Data *_data, const bool _prod_constraint, const std::string &_output_path, std::vector<std::vector<std::vector<int>>> _weights, const std::vector<int> &_indexes_examples):
                      CPModel_MaxClassification(_data, _prod_constraint, _output_path, _weights, _indexes_examples){

      }

      CPModel_MaxSum(Data *_data, const bool _prod_constraint, const std::string &_output_path, const std::string _input_file):
                      CPModel_MaxClassification(_data, _prod_constraint, _output_path, _input_file){

      }


      /* declare_sum_variable method
      Parameters : None
      Output : None
      */
      void declare_sum_variable(){
        sum_weights_example.resize(nb_examples);
        temp_objective.resize(nb_examples);
        for (size_t i = 0; i < nb_examples; i++) {
          sum_weights_example[i] = cp_model_builder.NewIntVar(Domain(-255*bnn_data->get_archi(0),255*bnn_data->get_archi(0)));
          temp_objective[i] = cp_model_builder.NewIntVar(Domain(-255*bnn_data->get_archi(0),255*bnn_data->get_archi(0)));
        }
      }

      /* model_preactivation_constraint method
      Parameters :
      - index_example : index of the example to classifie
      - l : layer \in [1, bnn_data->get_layers()-1]
      - j : neuron on layer l \in [0, bnn_data->get_archi(l)]
      Output : None
      Redefinition of the method in class cp_model in order to get the term of weights and
      activation values on the first layer instead of compute it again
      */
      void model_preactivation_constraint(const int &index_example, const int &l, const int &j){
        /*assert(index_example>=0);
        assert(index_example<nb_examples);
        assert(l>0);
        assert(l<bnn_data->get_layers());
        assert(j>=0);
        assert(j<bnn_data->get_archi(l));*/

        if(l == 1){
          LinearExpr temp(0);
          int tmp = bnn_data->get_archi(0);
          for (size_t i = 0; i < tmp; i++) {
            if (activation_first_layer[index_example][i] != 0) {
    					temp.AddTerm(get_w_ilj(i, l, j), activation_first_layer[index_example][i]);
    				}
          }
          cp_model_builder.AddEquality(sum_weights_example[index_example], temp);               //new instruction
          cp_model_builder.AddEquality(get_a_lj(index_example, 1, j), temp);
          }
        else{
          int tmp = bnn_data->get_archi(l-1);
          std::vector<IntVar> temp(tmp);
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

              cp_model_builder.AddEquality(temp[i], 0).OnlyEnforceIf(get_weight_is_0_ilj (i,l,j));
    					cp_model_builder.AddNotEqual(temp[i], 0).OnlyEnforceIf(Not(get_weight_is_0_ilj (i,l,j) ) );

              BoolVar b3 = cp_model_builder.NewBoolVar();

              // Implement b3 == (temp[i] == 1)
              cp_model_builder.AddEquality(temp[i], 1).OnlyEnforceIf(b3);
              cp_model_builder.AddNotEqual(temp[i], 1).OnlyEnforceIf(Not(b3));
              //Implement b3 == (weights == activation)
              cp_model_builder.AddEquality(get_w_ilj(i, l, j), activation[index_example][l-2][i]).OnlyEnforceIf(b3);
              cp_model_builder.AddNotEqual(get_w_ilj(i, l, j), activation[index_example][l-2][i]).OnlyEnforceIf(Not(b3));

            }
          }
          cp_model_builder.AddEquality(get_a_lj(index_example, l, j), LinearExpr::Sum(temp));
        }
      }

      /* model_objective_maximize_classification method
      This function sums all the values of classification in the LinearExpr objectif
      Parameters : None
      Output : None
      */
      void model_declare_objective(){
        for (size_t i = 0; i < nb_examples; i++) {
          cp_model_builder.AddAbsEquality(temp_objective[i], sum_weights_example[i]).OnlyEnforceIf(classification[i]);
          cp_model_builder.AddEquality(temp_objective[i], cp_model_builder.NewIntVar(Domain(0))).OnlyEnforceIf(Not(classification[i]));
          objectif.AddVar(temp_objective[i]);
        }
      }

      /* run method
      This function calls all the necessary methods to run the solver
      Parameters :
      - nb_seconds : Sets a time limit of nb_seconds
      Output : None
      */
      void run(const double &nb_seconds ,  std::string _strategy){
        declare_sum_variable();
        CPModel_MaxClassification::run(nb_seconds, _strategy);
      }

    };

  }
}

#endif /* EXAMPLES_CPP_CP_MAXSUM_WEIGHTS_EXAMPLE_MODEL_H_ */
