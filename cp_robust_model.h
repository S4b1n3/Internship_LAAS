#ifndef EXAMPLES_CPP_CP_ROBUST_MODEL_H_
#define EXAMPLES_CPP_CP_ROBUST_MODEL_H_

#include "ortools/sat/cp_model.h"
#include "ortools/sat/model.h"
#include "ortools/sat/sat_parameters.pb.h"

#include "ortools/util/sorted_interval_list.h"
#include "ortools/sat/cp_model_checker.h"

#include "cp_model.h"
#include "cp_minweight_model.h"


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
    class CPModel_Robust : public CPModel_MinWeight {
    private:
      std::vector<std::vector<BoolVar>> a;
      std::vector<std::vector<std::vector<IntVar>>> adversarial;
      int k;

    public:
      CPModel_Robust(Data *_data, const int &_nb_examples, const bool _prod_constraint, const std::string &_output_path, const int &_k):
                        CPModel_MinWeight(_data, _nb_examples, _prod_constraint, _output_path), k(_k){
      }

      CPModel_Robust(const int &_nb_examples, Data *_data, const bool _prod_constraint, const std::string &_output_path, const int &_k):
                        CPModel_MinWeight( _nb_examples,_data, _prod_constraint, _output_path), k(_k){
      }

      CPModel_Robust(Data *_data, const bool _prod_constraint, const std::string &_output_path, std::vector<std::vector<std::vector<int>>> _weights, const std::vector<int> &_indexes_examples, const int &_k):
                      CPModel_MinWeight(_data, _prod_constraint, _output_path, _weights, _indexes_examples), k(_k){

      }

      void declare_a_e_j_variables(){
        a.resize(nb_examples);
        int temp = bnn_data->get_archi(1);
        for (size_t e = 0; e < nb_examples; e++) {
          a[e].resize(temp);
          for (size_t j = 0; j < temp; j++) {
            a[e][j] = cp_model_builder.NewBoolVar();
          }
        }
      }

      void declare_adversarial_variables(){
        adversarial.resize(nb_examples);
        int temp = bnn_data->get_archi(1);
        int temp2 = bnn_data->get_archi(0);
        for (size_t e = 0; e < nb_examples; e++) {
          adversarial[e].resize(temp);
          for (size_t j = 0; j < temp; j++) {
            adversarial[e][j].resize(temp2);
            for (size_t i = 0; i < temp2; i++) {
              adversarial[e][j][i] = cp_model_builder.NewIntVar(Domain::FromValues({inputs[e][i]-k, 0, inputs[e][i]+k}));
            }
          }
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

          cp_model_builder.AddLessThan(tmp, 0).OnlyEnforceIf(a[index_example][j]);
      		cp_model_builder.AddGreaterOrEqual(tmp, 0).OnlyEnforceIf(Not(a[index_example][j]));


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

      void model_adversarial_variables(const int &index_example, const int &j){
        LinearExpr temp(0);

        int tmp = bnn_data->get_archi(0) ;
  			for (size_t i = 0; i < tmp; i++) {

          BoolVar b1 = cp_model_builder.NewBoolVar();
          BoolVar b2 = cp_model_builder.NewBoolVar();

          //Implement b1 == (weights > 0)
          cp_model_builder.AddGreaterThan(get_w_ilj(i, 1, j), 0).OnlyEnforceIf(b1);
          cp_model_builder.AddLessOrEqual(get_w_ilj(i, 1, j), 0).OnlyEnforceIf(Not(b1));

          //Implement b2 == (weights == 0)
          cp_model_builder.AddEquality(get_w_ilj(i, 1, j), 0).OnlyEnforceIf(b2);
          cp_model_builder.AddNotEqual(get_w_ilj(i, 1, j), 0).OnlyEnforceIf(Not(b2));

          //first constraint in the section of the note
          cp_model_builder.AddEquality(adversarial[index_example][j][i], 0).OnlyEnforceIf(b2);

          //following constraints in the section of the note
          cp_model_builder.AddEquality(adversarial[index_example][j][i], inputs[index_example][i]-k).OnlyEnforceIf({a[index_example][j], b1});
          cp_model_builder.AddEquality(adversarial[index_example][j][i], inputs[index_example][i]-k).OnlyEnforceIf({Not(a[index_example][j]), Not(b1)});

          cp_model_builder.AddEquality(adversarial[index_example][j][i], inputs[index_example][i]+k).OnlyEnforceIf({a[index_example][j], Not(b1)});
          cp_model_builder.AddEquality(adversarial[index_example][j][i], inputs[index_example][i]+k).OnlyEnforceIf({Not(a[index_example][j]), b1});

          temp.AddVar(adversarial[index_example][j][i]);
        }

        //two last constraints in the section of the note
        cp_model_builder.AddGreaterOrEqual(temp, 0).OnlyEnforceIf(a[index_example][j]);
        cp_model_builder.AddLessThan(temp, 0).OnlyEnforceIf(Not(a[index_example][j]));

      }

      /* run method
      This function calls all the necessary methods to run the solver
      Parameters :
      - nb_seconds : Sets a time limit of nb_seconds
      Output : None
      */
      void run(const double &nb_seconds , std::string _strategy){
        std::cout<< " c declare variables and constraints " <<std::endl;

    		std::clock_t c_start = std::clock();


    		assert(nb_seconds>0);
        declare_weight_variables();
    		activation_first_layer.resize(nb_examples);
    		activation.resize(nb_examples);
    		preactivation.resize(nb_examples);
    		for (size_t i = 0; i < nb_examples; i++) {
    			declare_preactivation_variables(i);
    			declare_activation_variables(i);
    		}
        declare_a_e_j_variables();
        declare_adversarial_variables();

        int temp = bnn_data->get_archi(1);
        for (size_t e = 0; e < nb_examples; e++) {
          for (size_t j = 0; j < temp; j++) {
            model_adversarial_variables(e, j);
          }
        }
        for (size_t i = 0; i < nb_examples; i++) {
    			int tmp = bnn_data->get_layers();
    			for (size_t l = 1; l < tmp; l++) {
    				int tmp2 =  bnn_data->get_archi(l);
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
    		std::ofstream parser(output_path.c_str(), std::ios::app);
    		parser << "d SETUP_TIME " << (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
    		parser.close();

    		std::cout << " c Setup finished; CPU setup time is " << (c_end-c_start) / CLOCKS_PER_SEC << " s" <<std::endl;
    		std::cout << "\n c Some statistics on the model : " << '\n';
    		LOG(INFO) << CpModelStats(cp_model_builder.Build());

    		std::cout<< " c running the solver.. " <<std::endl;
        cp_model_builder.Minimize(objectif);
      }






    };

  }
}
#endif /* EXAMPLES_CPP_CP_ROBUST_MODEL_H_ */
