#ifndef EXAMPLES_CPP_CP_MAXCLASSIFICATION_MODEL_H_
#define EXAMPLES_CPP_CP_MAXCLASSIFICATION_MODEL_H_

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

    class CPModel_MaxClassification : public CP_Model{
      protected :
        std::vector<BoolVar> classification;

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
      CPModel_MaxClassification(const std::vector<int> &_archi, const int &_nb_examples, const bool _prod_constraint, const std::string &_output_path):
                        CP_Model(_archi, _nb_examples, _prod_constraint, _output_path){

      }


      void declare_classification_variable(){
        classification.resize(nb_examples);
        for (size_t i = 0; i < nb_examples; i++) {
          classification[i] = cp_model.NewBoolVar();
        }
      }

      /* model_output_constraint method

         the example is well classified => activation[index_examples][bnn_data.get_layers()-2][label] == 1 and
                                           activation[index_examples][bnn_data.get_layers()-2][i] == -1

        classification[index_examples] = 1 => activation[index_examples][bnn_data.get_layers()-2][label] ==  1
        classification[index_examples] = 1 => last_layer == -9

      Parameters :
       - index_example : index of the example to classifie
      Output : None
      */
      void model_output_constraint(const int &index_examples){
        assert(index_examples >= 0);
        assert(index_example < nb_examples);

        LinearExpr last_layer(0);
        const int label = (int)bnn_data.get_dataset().training_labels[index_examples+index_rand];
        for (size_t i = 0; i < bnn_data.get_archi(bnn_data.get_layers()-1); i++) {
          if (i != label) {
            last_layer.AddVar(activation[index_examples][bnn_data.get_layers()-2][i]);
          }
        }

        cp_model.AddEquality(activation[index_examples][bnn_data.get_layers()-2][label], 1).OnlyEnforceIf(classification[index_examples]);
        cp_model.AddEquality(last_layer, -9).OnlyEnforceIf(classification[index_examples]);

      }


      /* model_objective_maximize_classification method
      This function sums all the values of classification in the LinearExpr objectif
      Parameters : None
      Output : None
      */
      void model_declare_objective(const int &index_example){
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
      void run(const double &nb_seconds){
        declare_classification_variable();
        CP_Model::run(nb_seconds);
        cp_model.Maximize(objectif);                        //objective function
      }


      void print_solution(const CpSolverResponse &r, const int &index = 0){
        assert(index >=0);
        if(r.status() == CpSolverStatus::OPTIMAL || r.status() == CpSolverStatus::FEASIBLE){
          //std::cout << "Activation 1 for neuron[label] : " << SolutionIntegerValue(r, a) << std::endl;
          //std::cout << "Activation -1 for other neurons : " << SolutionIntegerValue(r, b) << std::endl;
          for (size_t i = 0; i < nb_examples; i++) {
            std::cout << "classification[i] : " << SolutionIntegerValue(r, classification[i]) << std::endl;
          }

        }
        if(r.status()==CpSolverStatus::MODEL_INVALID){
          LOG(INFO) << ValidateCpModel(cp_model.Build());
        }
      }

    }; // close class CPModel

  } // close namespace sat
} // close namespace operations_research



#endif /* EXAMPLES_CPP_CP_MAXCLASSIFICATION_MODEL_H_ */
