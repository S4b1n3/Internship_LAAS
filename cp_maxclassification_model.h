#ifndef EXAMPLES_CPP_CP_MAXCLASSIFICATION_MODEL_H_
#define EXAMPLES_CPP_CP_MAXCLASSIFICATION_MODEL_H_

#include "ortools/sat/cp_model.h"
#include "ortools/sat/model.h"
#include "ortools/sat/sat_parameters.pb.h"

#include "ortools/util/sorted_interval_list.h"
#include "ortools/sat/cp_model_checker.h"


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

    class CPModel_MaxClassification {
      private :
        Data bnn_data;
        CpModelBuilder cp_model;
        int nb_examples;

        //weights[a][b][c] is the weight variable of the arc between neuron b on layer a-1 and neuron c on layer a
        std::vector<std::vector <std::vector<IntVar>>> weights;
        std::vector<std::vector <std::vector<int>>> weights_solution;

        std::vector<BoolVar> classification;

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
        bnn_data(_archi), domain(-1,1), activation_domain(Domain::FromValues({-1,1})), file_out("tests/solver_solution_"),
        file_out_extension(".tex"), nb_examples(_nb_examples), prod_constraint(_prod_constraint), output_path(_output_path){
        std::cout << "number of layers : "<<bnn_data.get_layers() << '\n';
        bnn_data.print_archi();
        bnn_data.print_dataset();
        index_rand = rand()%60000;
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

      std::vector<std::vector <std::vector<int>>> get_solution() const{
        return weights_solution;
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

        activation_first_layer.resize(nb_examples);
        activation_first_layer[index_example].resize(bnn_data.get_dataset().training_images[index_example+index_rand].size());
        for(size_t j = 0; j < bnn_data.get_archi(0); ++j){
          activation_first_layer[index_example][j] = (int64)bnn_data.get_dataset().training_images[index_example+index_rand][j];
        }
        activation.resize(nb_examples);
        activation[index_example].resize(bnn_data.get_layers()-1);
        for (size_t l = 0; l < bnn_data.get_layers()-1; ++l) {
          activation[index_example][l].resize(bnn_data.get_archi(l+1));
          for(size_t j = 0; j < bnn_data.get_archi(l+1); ++j){
            activation[index_example][l][j] = cp_model.NewIntVar(activation_domain);
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
        preactivation[index_example].resize(bnn_data.get_layers()-1);
        for (size_t l = 0; l < bnn_data.get_layers()-1; l++) {
          preactivation[index_example][l].resize(bnn_data.get_archi(l+1));
          for(size_t j = 0; j < bnn_data.get_archi(l+1); j++){
            preactivation[index_example][l][j] = cp_model.NewIntVar(domain);
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

        //Initialization of the variable

        weights.resize(bnn_data.get_layers());
        for (size_t l = 1; l < bnn_data.get_layers(); l++) {
          weights[l-1].resize(bnn_data.get_archi(l-1));
          for(size_t i = 0; i < bnn_data.get_archi(l-1); i++){
            weights[l-1][i].resize(bnn_data.get_archi(l));
            for (size_t j = 0; j < bnn_data.get_archi(l); j++) {

              /*One weight for each connection between the neurons i of layer
                l-1 and the neuron j of layer l : N(i) * N(i+1) connections*/

              weights[l-1][i][j] = cp_model.NewIntVar(domain);
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
        assert (index_example>=0);
        assert (index_example<nb_examples);
        assert (l>0);
        assert (l<bnn_data.get_layers());
        assert (j>=0);
        assert (j<bnn_data.get_archi(l));
          //_temp_bool is true iff preactivation[l][j] < 0
          //_temp_bool is false iff preactivation[l][j] >= 0
        const BoolVar _temp_bool = cp_model.NewBoolVar();
        cp_model.AddLessThan(get_a_lj(index_example, l, j), 0).OnlyEnforceIf(_temp_bool);
        cp_model.AddGreaterOrEqual(get_a_lj(index_example, l, j), 0).OnlyEnforceIf(Not(_temp_bool));
        cp_model.AddEquality(activation[index_example][l-1][j], -1).OnlyEnforceIf(_temp_bool);
        cp_model.AddEquality(activation[index_example][l-1][j], 1).OnlyEnforceIf(Not(_temp_bool));
      }

      /* model_preactivation_constraint method
      Parameters :
      - index_example : index of the example to classifie
      - l : layer \in [1, bnn_data.get_layers()-1]
      - j : neuron on layer l \in [0, bnn_data.get_archi(l)]
      Output : None
      */
      void model_preactivation_constraint(const int &index_example, const int &l, const int &j){
        assert(index_example>=0);
        assert(index_example<nb_examples);
        assert(l>0);
        assert(l<bnn_data.get_layers());
        assert(j>=0);
        assert(j<bnn_data.get_archi(l));
        if(l == 1){
          LinearExpr temp(0);
          for (size_t i = 0; i < bnn_data.get_archi(0); i++) {
            temp.AddTerm(get_w_ilj(i, l, j), activation_first_layer[index_example][i]);
          }
          cp_model.AddEquality(get_a_lj(index_example, 1, j), temp);
          }
        else{
          std::vector<IntVar> temp(bnn_data.get_archi(l-1));
          for (size_t i = 0; i < bnn_data.get_archi(l-1); i++) {
            temp[i] = cp_model.NewIntVar(domain);
            if(!prod_constraint){

              IntVar sum_weights_activation = cp_model.NewIntVar(Domain(-2,2));
              IntVar sum_temp_1 = cp_model.NewIntVar(Domain(0, 2));
              cp_model.AddEquality(sum_weights_activation, LinearExpr::Sum({get_w_ilj(i, l, j), activation[index_example][l-2][i]}));
              cp_model.AddEquality(sum_temp_1, temp[i].AddConstant(1));
              cp_model.AddAbsEquality(sum_temp_1, sum_weights_activation);

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

              BoolVar b1 = cp_model.NewBoolVar();
              BoolVar b2 = cp_model.NewBoolVar();

              // Implement b1 == (temp[i] == 0)
              cp_model.AddEquality(temp[i], 0).OnlyEnforceIf(b1);
              cp_model.AddNotEqual(temp[i], LinearExpr(0)).OnlyEnforceIf(Not(b1));
              //Implement b2 == (weights == 0)
              cp_model.AddEquality(get_w_ilj(i, l, j), 0).OnlyEnforceIf(b2);
              cp_model.AddNotEqual(get_w_ilj(i, l, j), LinearExpr(0)).OnlyEnforceIf(Not(b2));

              // b1 implies b2 and b2 implies b1
              cp_model.AddImplication(b2, b1);
              cp_model.AddImplication(b1, b2);

              BoolVar b3 = cp_model.NewBoolVar();
              BoolVar b4 = cp_model.NewBoolVar();

              // Implement b3 == (temp[i] == 1)
              cp_model.AddEquality(temp[i], 1).OnlyEnforceIf(b3);
              cp_model.AddNotEqual(temp[i], LinearExpr(1)).OnlyEnforceIf(Not(b3));
              //Implement b4 == (weights == activation)
              cp_model.AddEquality(get_w_ilj(i, l, j), activation[index_example][l-2][i]).OnlyEnforceIf(b4);
              cp_model.AddNotEqual(get_w_ilj(i, l, j), activation[index_example][l-2][i]).OnlyEnforceIf(Not(b4));


              // b3 implies b4 and b4 implies b3
              cp_model.AddImplication(b3, b4);
              cp_model.AddImplication(b4, b3);

            }
          }
          cp_model.AddEquality(get_a_lj(index_example, l, j), LinearExpr::Sum(temp));
        }
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
      void model_objective_maximize_classification(){
        for (size_t i = 0; i < nb_examples; i++) {
          objectif.AddVar(classification[i]);
        }
      }

      /* run method
      This function calls all the necessary methods to run the solver
      Parameters :
      - nb_seconds : Sets a time limit of nb_seconds
      - nb_examples : number of examples
      Output : None
      */
      void run(const double &nb_seconds){
        assert(nb_seconds>0);
        declare_weight_variables();
        //initialization of the variables
        declare_classification_variable();
        for (size_t i = 0; i < nb_examples; i++) {
          declare_preactivation_variables(i);
          declare_activation_variables(i);
        }
        for (size_t i = 0; i < nb_examples; i++) {
          for (size_t l = 1; l < bnn_data.get_layers(); l++) {
            for (size_t j = 0; j < bnn_data.get_archi(l); j++) {
              model_preactivation_constraint(i, l, j);
              model_activation_constraint(i, l, j);
            }
          }
        }
        for (size_t i = 0; i < nb_examples; i++) {
          model_output_constraint(i);
        }
        model_objective_maximize_classification();                //initialization of the objective
        parameters.set_max_time_in_seconds(nb_seconds);     //Add a timelimit
        model.Add(NewSatParameters(parameters));
        cp_model.Maximize(objectif);                        //objective function

      }


      void check(const CpSolverResponse &r, const int &index=0){
        std::string solution_file = output_path+"/solution"+std::to_string(nb_examples)+".txt";
        std::ofstream solution(solution_file.c_str(), std::ios::app);

        solution << "Weights : " << std::endl;
        weights_solution.resize(bnn_data.get_layers());
        for (size_t l = 1; l < bnn_data.get_layers(); ++l) {
          solution << "#Layer "<< l << std::endl;
          weights_solution[l-1].resize(bnn_data.get_archi(l-1));
          for (size_t i = 0; i < bnn_data.get_archi(l-1); ++i) {
            solution << "#Neuron " << i << std::endl;
            weights_solution[l-1][i].resize(bnn_data.get_archi(l));
            for (size_t j = 0; j < bnn_data.get_archi(l); ++j) {
              weights_solution[l-1][i][j] = SolutionIntegerValue(r, weights[l-1][i][j]);
              solution << weights_solution[l-1][i][j] << "  " ;
            }
            solution << std::endl;
          }
          solution << std::endl;
        }

        for (size_t i = 0; i < nb_examples; i++) {
          const int label = (int)bnn_data.get_dataset().training_labels[i+index_rand];
          solution << "#Example " << i << "(label " << label << ")" << std::endl;
          solution << "#Preactivation values" << std::endl;
          preactivation_solution.resize(bnn_data.get_layers()-1);
          for (size_t l = 0; l < bnn_data.get_layers()-1; l++) {
            solution << "#Layer " << l+1 << std::endl;
            preactivation_solution[l].resize(bnn_data.get_archi(l+1));
            for(size_t j = 0; j < bnn_data.get_archi(l+1); j++){
              preactivation_solution[l][j] = SolutionIntegerValue(r, preactivation[i][l][j]);
              solution << preactivation_solution[l][j] << "  ";
            }
            solution << std::endl;
          }

          solution << "#Activation values" << std::endl;
          activation_solution.resize(bnn_data.get_layers());
          for (size_t l = 0; l < bnn_data.get_layers(); l++) {
            solution << "#Layer " << l << std::endl;
            activation_solution[l].resize(bnn_data.get_archi(l));
            for(size_t j = 0; j < bnn_data.get_archi(l); j++){
              if(l == 0){
                activation_solution[l][j] = (int)activation_first_layer[i][j];
              }
              else{
                activation_solution[l][j] = SolutionIntegerValue(r, activation[i][l-1][j]);
              }
              solution << activation_solution[l][j] << "  ";
            }
            solution << std::endl;
          }


          Solution check_solution(bnn_data.get_archi(), weights_solution, activation_solution, preactivation_solution, i+index_rand);
          std::cout << "Checking solution : "<<index<<" : ";
          bool checking = check_solution.run_solution();
        }
      }

      // Print some statistics from the solver: Runtime, number of nodes, number of propagation (filtering, pruning), memory,
      // Status: Optimal, suboptimal, satisfiable, unsatisfiable, unkown
      // Output Status: {OPTIMAL, FEASIBLE, INFEASIBLE, MODEL_INVALID, UNKNOWN}
      void print_statistics(){
        response = SolveCpModel(cp_model.Build(), &model);
        std::string result_file = output_path+"/results"+std::to_string(nb_examples)+".stat";
        std::ofstream parser(result_file.c_str(), std::ios::app);
        std::cout << "\nSome statistics on the solver response : " << '\n';
        LOG(INFO) << CpSolverResponseStats(response);
        std::cout << "\nSome statistics on the model : " << '\n';
        LOG(INFO) << CpModelStats(cp_model.Build());
        if(parser){
          parser << std::endl << "run time " << response.wall_time() << std::endl;
          parser << "memory " << sysinfo::MemoryUsageProcess() << std::endl;
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
          parser << CpModelStats(cp_model.Build()) << std::endl;
          parser << std::endl;
        }
        else
          std::cout << "Error opening parser file" << '\n';
        if (response.status()== CpSolverStatus::OPTIMAL || response.status() == CpSolverStatus::FEASIBLE) {
          check(response);
        }
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
