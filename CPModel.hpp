#ifndef DEF_CPMODEL
#define DEF_CPMODEL

#include "Data.hpp"
#include "Solution.hpp"

#include "ortools/sat/cp_model.h"
#include "ortools/sat/model.h"
#include "ortools/sat/sat_parameters.pb.h"
#include "ortools/util/sorted_interval_list.h"
#include "ortools/sat/cp_model_checker.h"



namespace operations_research{
  namespace sat{

    /* Class Model contains the constraint programming problem
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
    */
    class CPModel {

    private:

      Data bnn_data;
      CpModelBuilder cp_model;
      int nb_examples;

      //weights[a][b][c] is the weight variable of the arc between neuron b on layer a-1 and neuron c on layer a
      std::vector<std::vector <std::vector<IntVar>>> weights;
      std::vector<std::vector <std::vector<int>>> solution;

      std::vector <std::vector<std::vector<IntVar>>> activation;
      std::vector <std::vector<std::vector<IntVar>>> preactivation;
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


    public:

      /*
      Constructor of the class CPModel
      Argument :
      - a vector representing the architecture of a BNN
      - nb_examples : number of examples to test
      The constructor initialize the data of the problem and the domain of the variables
      Call the constructor launch the method to solve the problem
      */
      CPModel(const std::vector<int> &_archi, const int &_nb_examples);

      /* Getters */

      //returns the data of the problem
      Data get_data() const;

      //return the response of the problem
      CpSolverResponse get_response() const;

      std::vector<std::vector <std::vector<int>>> get_solution() const;

      /* declare_activation_variable method
      Parameters :
      - index_example : index of the training example to classifie
      Output : None
      n_{lj} variables from the CP paper
      */
      void declare_activation_variables(const int &index_example);


      /* declare_preactivation_variable method
      Parameters :
      - index_example : index of the training example to classifie
      Output : None
      preactivation[l] represents the preactivation of layer l+1 where l \in [0,bnn_data.get_layers()-1]
      */
      void declare_preactivation_variables(const int &index_example);

      /* get_a_lj method
      Parameters :
      - index_example : index of the example to classifie
      - l : layer \in [1, bnn_data.get_layers()]
      - j : neuron on layer l \in [0, bnn_data.get_archi(l)]
      Output :
      a_{lj} variables from the CP paper
      */
      IntVar get_a_lj(const int &index_example, const int &l, const int &j);


      /* declare_weight_variables method
      This method initialize the weight variables
      weights[a][b][c] is the weight variable of the edge between neuron b on layer a-1 and neuron c on layer a
      Parameters : None
      Output : None
      */
      void declare_weight_variables();


      /* get_w_ilj method
      Parameters :
      - i : neuron on layer l-1 \in [0, bnn_data.get_archi(l-1)]
      - l : layer \in [1, bnn_data.get_layers()]
      - j : neuron on layer l \in [0, bnn_data.get_archi(l)]
      Output :
      w_{ilj} variables from the CP paper
      */
      IntVar get_w_ilj(const int &i, const int &l, const int &j);


      /* model_objective_minimize_weight method
      This function sums all the weights in the LinearExpr objectif
      Parameters : None
      Output : None
      */
      void model_objective_minimize_weight();

      /* model_activation_constraint method
      Parameters :
      - index_example : index of the example to classifie
      - l : layer \in [1, bnn_data.get_layers()]
      - j : neuron on layer l \in [0, bnn_data.get_archi(l)]

      preactivation[l][j] >= 0 => activation[l][j] = 1
      preactivation[l][j] < 0 => activation[l][j] = -1
      Output : None
      */
      void model_activation_constraint(const int &index_example, const int &l, const int &j);

      /* model_preactivation_constraint method
      Parameters :
      - index_example : index of the example to classifie
      - l : layer \in [1, bnn_data.get_layers()-1]
      - j : neuron on layer l \in [0, bnn_data.get_archi(l)]
      Output : None
      */
      void model_preactivation_constraint(const int &index_example, const int &l, const int &j);


      /* model_output_constraint method
      This function forces the output to match the label
      Parameters :
      - index_examples : index of examples
      Output : None
      */
      void model_output_constraint(const int &index_examples);


      /* run method
      This function calls all the necessary methods to run the solver
      Parameters :
      - nb_seconds : Sets a time limit of nb_seconds
      - nb_examples : number of examples
      Output : None
      */
      void run(const double &nb_seconds);


      /*print_header_solution method
      This function writes on the output file the latex header
      Parameters :
      - num_sol : the index of the solution
      Output ; None
      */
      void print_header_solution(const int &num_sol);

      /* print_node mehod
      This function creates a node in latex
      Parameters :
      - name : name of the node
      - x, y : position of the node
      Output : a string containing the latex command that will create the node
      */
      std::string print_node(const std::string &name, const int &x, const int &y);

      /* print_arc mehod
      This function creates an arc in latex
      Parameters :
      - origin : origin node of the arc
      - target : target node of the arc
      - weight : value of the weight for this arc
      Output : a string containing the latex command that will create the arc
      */
      std::string print_arc(const std::string &origin, const std::string &target, const int &weight);

      // Print some statistics from the solver: Runtime, number of nodes, number of propagation (filtering, pruning), memory,
      // Status: Optimal, suboptimal, satisfiable, unsatisfiable, unkown
      // Output Status: {OPTIMAL, FEASIBLE, INFEASIBLE, MODEL_INVALID, UNKNOWN}
      void print_statistics();



      void print_solution(const CpSolverResponse &r, const int &index = 0);

      /* print_solution method
      This function prints a solution returned by the solver
      if this solution is feasible or optimal
      Parameters :
      - r : response of the solver
      - index : index of the solution (default : 0)
      Output : None
      */
      void print_solution_bis(const CpSolverResponse &r, const int &index = 0);

      /* print_all_solutions method
      This function prints all feasible or optimal solutions returned by the solver
      Parameters : None
      Output : None
      */
      void print_all_solutions();

    } ; //close class CPModel
  } //close namespace sat
} //close namespace operations_research

#endif
