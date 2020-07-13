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
  std::string output_file;

public:

  /* Constructor of the class Evaluation
  Arguments :
    - size : number of examples to test
    - _weights : values of the weights returned by the solver
    - archi : architecture of the network
  */
  Evaluation(const std::vector<std::vector<std::vector<int>>> &_weights, Data* model_data, const std::string &filename) :
            weights(_weights), checker(model_data, _weights), output_file(filename){
  }

  /* run_evaluation method
  Arguments :
    - test_set : boolean that indicate on which dataset the examples are taken (default : testing set)
  This method tests every example one by one from the dataset and return the accuracy
  */
  double run_evaluation (const int &test_set, const bool &predict){
    std::clock_t c_start = std::clock();
    if (test_set) {
nb_correct_classifications=0;
      for (size_t i = 0; i < 10000; i++) {
        if (checker.run_solution(false, false, true, predict, true, i)) {
          nb_correct_classifications += 1;
        }
      }
      std::clock_t c_end = std::clock();
      std::cout << " Evaluation on testing set finished; CPU setup time is " << (c_end-c_start) / CLOCKS_PER_SEC << " s" <<std::endl;
      std::ofstream parser(output_file.c_str(), std::ios::app);
  		parser << "test accuray time " << (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
  		parser.close();
      return 100*nb_correct_classifications/10000;
    }
    else{
nb_correct_classifications=0;
      for (size_t i = 0; i < 60000; i++) {
        if (checker.run_solution(false, false, true, predict, false, i)) {
          nb_correct_classifications += 1;
        }
      }
      std::clock_t c_end = std::clock();
      std::cout << " Evaluation on training set finished; CPU setup time is " << (c_end-c_start) / CLOCKS_PER_SEC << " s" <<std::endl;
      std::ofstream parser(output_file.c_str(), std::ios::app);
  		parser << "train accuray time " << (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
  		parser.close();
      return 100*nb_correct_classifications/10000;
    }
  }


};

#endif /* EXAMPLES_CPP_EVALUATION_H_ */
