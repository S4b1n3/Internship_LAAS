#ifndef EXAMPLES_CPP_EVALUATION_H_
#define EXAMPLES_CPP_EVALUATION_H_

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
  int nb_correct_classifications = 0;
  Solution checker;
  std::string output_file;

public:

  /* Constructor of the class Evaluation
  Arguments :
    - size : number of examples to test
    - _weights : values of the weights returned by the solver
    - archi : architecture of the network
  */
  Evaluation(std::vector<std::vector<std::vector<int>>> _weights, Data* model_data, const std::string &filename) :
            checker(model_data, _weights), output_file(filename){
  }

  Evaluation(std::vector<std::vector<std::vector<int>>> _weights, Data* model_data) :
            checker(model_data, _weights){
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
      checker.set_evaluation_config(false, false, true, predict, true);
      for (size_t i = 0; i < 10000; i++) {
    	  //if ( ! (i % 100) )
    	  //  std::cout<<"  c i "<< i << std::endl;
    	  //if (checker.run_solution(false, false, true, true, predict, true, i)) {
    	  if (checker.run_solution_light(i) )
    		  nb_correct_classifications += 1;
      }

      std::clock_t c_end = std::clock();
      std::cout << " c Evaluation finished; CPU setup time is " << (c_end-c_start) / CLOCKS_PER_SEC << " s ";
      std::ofstream parser(output_file.c_str(), std::ios::app);
  		parser << "d TEST_ACCURACY_TIME " << (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
  		parser.close();
      std::cout << " and accuracy value is "<< 100*nb_correct_classifications/10000 << '\n';
      return 100*nb_correct_classifications/10000;
    }
    else{
        nb_correct_classifications=0;
        checker.set_evaluation_config(false, false, true, predict, false);
        for (size_t i = 0; i < 60000; i++) {
      	  //if ( ! (i % 100) )
      	   // std::cout<<"  c i "<< i << std::endl;
      	  //if (checker.run_solution(false, false, true, true, predict, true, i)) {
      	  if (checker.run_solution_light(i) )
      		  nb_correct_classifications += 1;
        }

    	/*
      nb_correct_classifications=0;
      for (size_t i = 0; i < 60000; i++) {
        if (checker.run_solution(false, false, true, true, predict, false, i)) {
          nb_correct_classifications += 1;
        }
      }
      */
      std::clock_t c_end = std::clock();
      std::cout << " c Evaluation finished; CPU setup time is " << (c_end-c_start) / CLOCKS_PER_SEC << " s ";
      std::ofstream parser(output_file.c_str(), std::ios::app);
  		parser << "d TRAIN_ACCURACY_TIME " << (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
  		parser.close();
      std::cout << " and accuracy value is "<< 100*nb_correct_classifications/60000 << '\n';
      return 100*nb_correct_classifications/60000;
    }
  }

  std::vector<int> get_correct_examples(const bool &test_set = true){
    std::vector<int> indexes;
    if (test_set) {
      for (int i = 0; i < 10000; i++) {
        if (checker.run_solution(false, false, true, true, true, true, i)) {
          indexes.push_back(i);
        }
      }
    } else {
      for (int i = 0; i < 60000; i++) {
        if (checker.run_solution(false, false, true, true, true, false, i)) {
          indexes.push_back(i);
        }
      }
    }

    return indexes;
  }


};

#endif /* EXAMPLES_CPP_EVALUATION_H_ */
