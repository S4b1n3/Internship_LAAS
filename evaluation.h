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
  Data bnn_data;

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
  double run_evaluation (const int &test_set, const bool &strong_classification){
    std::clock_t c_start = std::clock();
    if (test_set) {
      nb_correct_classifications=0;
      checker.set_evaluation_config(false, false, true, strong_classification, true);
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

      std::cout << " ; nb_correct_classifications is "<< nb_correct_classifications ;

      std::cout << " and accuracy value is "<< 100*nb_correct_classifications/10000 << '\n';
      return 100*nb_correct_classifications/10000;
    }
    else{
        nb_correct_classifications=0;
        checker.set_evaluation_config(false, false, true, strong_classification, false);
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

      std::cout << " ; nb_correct_classifications is "<< nb_correct_classifications ;

      std::cout << " and accuracy value is "<< 100*nb_correct_classifications/60000 << '\n';
      return 100*nb_correct_classifications/60000;
    }
  }

  std::vector<int> get_correct_examples(const bool &test_set = true){
    std::vector<int> indexes;
    checker.set_evaluation_config(false, false, true, true, test_set);
    if (test_set) {
      std::vector<int> occ(10, 0);
      for (int i = 0; i < 10000; i++) {
        int label = (int)bnn_data.get_dataset().test_labels[i];
        if (checker.run_solution_light(i) && occ[label] == 0) {
          indexes.push_back(i);
          occ[label]++;
        }
      }
    } else {
      std::vector<int> occ(10, 0);
      for (int i = 0; i < 60000; i++) {
        int label = (int)bnn_data.get_dataset().training_labels[i];
        if (checker.run_solution_light(i) && occ[label] == 0) {
          indexes.push_back(i);
          occ[label]++;
        }
      }
    }

    return indexes;
  }


};

#endif /* EXAMPLES_CPP_EVALUATION_H_ */
