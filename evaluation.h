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
  Evaluation(const std::vector<std::vector<std::vector<int>>> &_weights, const Data &model_data) :
            weights(_weights), checker(model_data, weights){
  }

  /* run_evaluation method
  Arguments :
    - test_set : boolean that indicate on which dataset the examples are taken (default : testing set)
  This method tests every example one by one from the dataset and return the accuracy
  */
  double run_evaluation (const int &test_set = true){
    std::clock_t c_start = std::clock();
    if (test_set) {
      for (size_t i = 0; i < 10000; i++) {
        if (checker.run_solution(false, i, true, false)) {
          nb_correct_classifications += 1;
        }
      }
      std::clock_t c_end = std::clock();
      std::cout << " Evaluation on testing set finished; CPU setup time is " << (c_end-c_start) / CLOCKS_PER_SEC << " s" <<std::endl;
      return 100*nb_correct_classifications/10000;
    }
    else{
      for (size_t i = 0; i < 60000; i++) {
        if (checker.run_solution(false, i, false, false)) {
          nb_correct_classifications += 1;
        }
      }
      std::clock_t c_end = std::clock();
      std::cout << " Evaluation on training set finished; CPU setup time is " << (c_end-c_start) / CLOCKS_PER_SEC << " s" <<std::endl;
      return 100*nb_correct_classifications/10000;
    }
  }


};

#endif /* EXAMPLES_CPP_EVALUATION_H_ */
