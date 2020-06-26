#ifndef EXAMPLES_CPP_EVALUATION_H_
#define EXAMPLES_CPP_EVALUATION_H_

#include "cp_model.h"
#include "solution.h"


/* Class Evaluation used to mesure the accuray of the model found by the solver
Attributs :
  - weights : values of the weights returned by the solver
  - dataset_size : number of examples to test
  - nb_correct_classification : number of examples that are correctly classified
  - architecture : architecture of the network
  - checker : checker for one example from the class solution
*/
class Evaluation {
private:

  std::vector<std::vector<std::vector<int>>> weights;
  int dataset_size;
  int nb_correct_classifications = 0;
  std::vector<int> architecture;
  Solution checker;

public:

  /* Constructor of the class Evaluation
  Arguments :
    - size : number of examples to test
    - _weights : values of the weights returned by the solver
    - archi : architecture of the network
  */
  Evaluation(const int &size, const std::vector<std::vector<std::vector<int>>> &_weights, const Data &model_data) :
            weights(_weights), dataset_size(size), checker(model_data, weights){
  }

  /* run_evaluation method
  Arguments :
    - test_set : boolean that indicate on which dataset the examples are taken (default : testing set)
  This method tests every example one by one from the dataset and return the accuracy
  */
  double run_evaluation (const int &test_set = true){
    for (size_t i = 0; i < dataset_size; i++) {
      if (checker.run_solution(false, i, test_set, false)) {
        nb_correct_classifications += 1;
      }
    }
    return 100*nb_correct_classifications/dataset_size;
  }


};

#endif /* EXAMPLES_CPP_EVALUATION_H_ */
