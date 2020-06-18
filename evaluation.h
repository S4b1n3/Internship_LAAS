#ifndef EXAMPLES_CPP_EVALUATION_H_
#define EXAMPLES_CPP_EVALUATION_H_

#include "cp_model.h"
#include "solution.h"



class Evaluation {
private:

  std::vector<std::vector<std::vector<int>>> weights;
  int test_set_size;
  int nb_correct_classification = 0;
  std::vector<int> architecture;

public:

  Evaluation(const int &size, std::vector<std::vector<std::vector<int>>> _weights, const std::vector<int> &archi) :
            weights(std::move(_weights)), test_set_size(size), architecture(archi){

  }

  double run_evaluation_test_set(){
    for (size_t i = 0; i < test_set_size; i++) {
      bool classification;
      Solution checker(architecture, weights, i, true);
      classification = checker.run_solution(false);
      if (classification) {
        nb_correct_classification += 1;
      }
    }
  }

  double run_evaluation_train_set(){
    for (size_t i = 0; i < test_set_size; i++) {
      bool classification;
      Solution checker(architecture, weights, i, false);
      classification = checker.run_solution(false);
      if (classification) {
        nb_correct_classification += 1;
      }
    }

  return 100*nb_correct_classification/test_set_size;
}


};

#endif /* EXAMPLES_CPP_EVALUATION_H_ */
