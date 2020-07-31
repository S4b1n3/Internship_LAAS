#ifndef EXAMPLES_CPP_CP_SATISFACTION_MODEL_H_
#define EXAMPLES_CPP_CP_SATISFACTION_MODEL_H_

#include "ortools/sat/cp_model.h"
#include "ortools/sat/model.h"
#include "ortools/sat/sat_parameters.pb.h"

#include "ortools/util/sorted_interval_list.h"
#include "ortools/sat/cp_model_checker.h"

#include <memory>
#include "ortools/port/sysinfo.h"

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

    /* CPModel_MinWeight contains the constraint programming model for the full classification and minweight problem
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
    class CPModel_Satisfaction : public CP_Model{

    private:
      std::vector<std::vector<IntVar>> sum;

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
      CPModel_Satisfaction(Data *_data, const int &_nb_examples, const bool _prod_constraint, const std::string &_output_path):
                        CP_Model(_data, _nb_examples, _prod_constraint, _output_path){

      }

      CPModel_Satisfaction(const int &_nb_examples_per_label, Data *_data, const bool _prod_constraint, const std::string &_output_path):
                        CP_Model(_nb_examples_per_label, _data, _prod_constraint, _output_path){

      }

      CPModel_Satisfaction(Data *_data, const bool _prod_constraint, const std::string &_output_path, std::vector<std::vector<std::vector<int>>> _weights, const std::vector<int> &_indexes_examples):
                      CP_Model(_data, _prod_constraint, _output_path, _weights, _indexes_examples){

      }

      CPModel_Satisfaction(Data *_data, const bool _prod_constraint, const std::string &_output_path, const std::string &_input_file):
                      CP_Model(_data, _prod_constraint, _output_path, _input_file){

      }

      CPModel_Satisfaction(Data *_data, const bool _prod_constraint, const std::string &_output_path, const std::string &_input_file, const std::string &_solution_file):
                      CP_Model(_data, _prod_constraint, _output_path, _input_file, _solution_file){

      }


      /* model_objective_minimize_weight method
      This function sums all the weights in the LinearExpr objectif
      Parameters : None
      Output : None
      */
      void model_declare_objective(){
        int tmp = bnn_data->get_layers();
        for (size_t l = 1; l < tmp; l++) {
          int tmp2 = bnn_data->get_archi(l-1);
          for(size_t i = 0; i < tmp2; i++) {
            int tmp3 = bnn_data->get_archi(l);
            for (size_t j = 0; j < tmp3; j++) {
              IntVar abs = cp_model_builder.NewIntVar(Domain(0,1));
              cp_model_builder.AddAbsEquality(abs, weights[l-1][i][j]);
              objectif.AddVar(abs);
            }
          }
        }
      }

      /* model_output_constraint method
      This function forces the output to match the label
      Parameters :
      - index_examples : index of examples
      Output : None
      */
      void model_output_constraint(const int &index_examples){
        /*assert(index_examples >= 0);
        assert(index_example < nb_examples);*/
        const int label = labels[index_examples];
        int tmp2 = bnn_data->get_layers()-2;
        cp_model_builder.AddEquality(activation[index_examples][tmp2][label], 1);
        int tmp = bnn_data->get_archi(bnn_data->get_layers()-1);
        for (size_t i = 0; i < tmp; i++) {
          if (i != label) {
            cp_model_builder.AddEquality(activation[index_examples][tmp2][i], -1);
          }
        }
      }


      void print_solution(const CpSolverResponse &r, const int &index = 0){
        assert(index >=0);
        if(r.status() == CpSolverStatus::OPTIMAL || r.status() == CpSolverStatus::FEASIBLE){
          std::cout << "\n s Solution "<< index << " : \n";
          for (size_t l = 1; l < bnn_data->get_layers(); ++l) {
            std::cout << " s Layer "<< l << ": \n";
            for (size_t i = 0; i < bnn_data->get_archi(l-1); ++i) {
              for (size_t j = 0; j < bnn_data->get_archi(l); ++j) {
                std::cout << "\t s w["<<l<<"]["<<i<<"]["<<j<<"] = " <<SolutionIntegerValue(r, weights[l-1][i][j]);
              }
              std::cout << '\n';
            }
            std::cout << '\n';
          }

          for (size_t l = 0; l < bnn_data->get_layers()-1; l++) {
            for(size_t j = 0; j < bnn_data->get_archi(l+1); j++){
              std::cout << " s preactivation["<<l<<"]["<<j<<"] = " << SolutionIntegerValue(r,preactivation[0][l][j])<<std::endl;
            }
          }

          for(size_t j = 0; j < bnn_data->get_archi(0); ++j){
            std::cout << " s activation_first_layer["<<j<<"] = " << SolutionIntegerValue(r,activation_first_layer[0][j]) << std::endl;
          }
          for (size_t l = 0; l < bnn_data->get_layers()-1; ++l) {
            for(size_t j = 0; j < bnn_data->get_archi(l+1); ++j){
              std::cout << " s activation["<<l<<"]["<<j<<"] = " << SolutionIntegerValue(r,activation[0][l][j])<<std::endl;
            }
          }

        }
        if(r.status()==CpSolverStatus::MODEL_INVALID){
          LOG(INFO) << ValidateCpModel(cp_model_builder.Build());
        }
      }

    } ; //close class CPModel
  } //close namespace sat
} //close namespace operations_research






#endif /* EXAMPLES_CPP_CP_MINWEIGHT_MODEL_H_ */