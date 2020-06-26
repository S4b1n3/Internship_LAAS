
/*
 * solution.h
 *
 *  Created on: Jun 7, 2020
 *      Author: msiala
 */



#ifndef EXAMPLES_CPP_SOLUTION_H_
#define EXAMPLES_CPP_SOLUTION_H_

#include <ctime>



/* Class solution used to check that the solution returned by the solver is correct
Attributs :
- bnn_data : data of the network
- weights : values of the weights that have to be tested
- nb_layers : number of layer in the network
- preactivation : preactivation values for each neuron
- activation : activation values for each neuron
- solver_preactivation : value of the preactivation of each neuron returned by the solver
- solver_activation : value of the activation of each neuron returned by the solver
- example_images : inputs of the network
- example_label : labels of the inputs (the output must be equal to the label)
*/
class Solution {

private:
  Data bnn_data;
  std::vector<std::vector<std::vector<int>>> weights;
  int nb_layers;
  std::vector<std::vector<int>> preactivation;
  std::vector<std::vector<int>> activation;
  std::vector<std::vector<int>> solver_preactivation;
  std::vector<std::vector<int>> solver_activation;
  std::vector<uint8_t> example_images;
  int example_label;

public:

  /* Constructors of the class Solution
  Arguments :
    - archi : architecture of the network
    - _weights : solution that has to be tested
    - _preactivation : preactivation values for each neuron returned by the solver
    - _activation : activation values for each neuron returned by the solver
    - index_example : index of the input in the training set
    - test_set : booean that indicates which dataset to use for the tests
  */
  Solution(const Data &model_data, std::vector<std::vector<std::vector<int>>> _weights, std::vector<std::vector<int>> _activation, std::vector<std::vector<int>> _preactivation, const int &index_example):
  bnn_data(model_data), weights(std::move(_weights)), solver_activation(std::move(_activation)), solver_preactivation(std::move(_preactivation)){
    nb_layers = bnn_data.get_layers();
    example_label = (int)bnn_data.get_dataset().training_labels[index_example];
    example_images = bnn_data.get_dataset().training_images[index_example];
  }

  Solution(const Data &model_data, std::vector<std::vector<std::vector<int>>> _weights, const int &index_example):
  bnn_data(model_data), weights(std::move(_weights)){
    nb_layers = bnn_data.get_layers();
    example_label = (int)bnn_data.get_dataset().training_labels[index_example];
    example_images = bnn_data.get_dataset().training_images[index_example];
  }


  Solution(const Data &model_data, std::vector<std::vector<std::vector<int>>> _weights, const int &index_example, const bool test_set):
  bnn_data(model_data), weights(std::move(_weights)){
    nb_layers = bnn_data.get_layers();
    if(test_set){
      example_label = (int)bnn_data.get_dataset().test_labels[index_example];
      example_images = bnn_data.get_dataset().test_images[index_example];
    }
    else{
      example_label = (int)bnn_data.get_dataset().training_labels[index_example];
      example_images = bnn_data.get_dataset().training_images[index_example];
    }
  }

  Solution(const Data &model_data, std::vector<std::vector<std::vector<int>>> _weights):
      bnn_data(model_data), weights(std::move(_weights)){
    nb_layers = bnn_data.get_layers();
  }



  /* activation_function method
  Given an preactivation x, returns 1 if the preactivation is positive and -1 either
  */
  static inline int activation_function(int x){
    if (x >= 0) return 1;
    else return -1;
  }

  /* init method
  This function initialize the preactivation and the activation of each neuron of the network
  Parameters : None
  Output : None
  */
  void init(const int &test_mode){
    //preactivation[i][j] is the value of the preactivation of the neuron j on layer i

    preactivation.resize(nb_layers);
    for (size_t l = 0; l < nb_layers; l++) {
      preactivation[l].resize(bnn_data.get_archi(l));
      for(size_t j = 0; j < bnn_data.get_archi(l); j++){
        preactivation[l][j] = 0;
      }
    }

    activation.resize(nb_layers);
     //activation[i][j] is the value of the activation of the neuron j on layer i
    for (size_t l = 0; l < nb_layers ; l++) {
      activation[l].resize(bnn_data.get_archi(l));
      for (size_t i = 0; i < bnn_data.get_archi(l); i++) {
        if(l == 0 && !test_mode){
          activation[l][i] = (int)example_images[i];
        }
        else{
          activation[l][i] = 0;
        }
      }
    }
  }


  /* predict method
  This function tests if the output of the network with the weights returned
  by the solver corresponds the label of the input
  Parameters : None
  Output : boolean -> true if the input is well classified and false either
  */
  bool predict(){
    bool result = true ;
    for (size_t l = 1; l < nb_layers; l++) {
      for (size_t i = 0; i < bnn_data.get_archi(l-1); i++) {
        for (size_t j = 0; j < bnn_data.get_archi(l); j++) {
          preactivation[l][j] += activation[l-1][i] * weights[l-1][i][j];
        }
        for (size_t j = 0; j < bnn_data.get_archi(l); j++) {
          activation[l][j] = activation_function(preactivation[l][j]);
        }
      }
    }
    int predict = 0, compt = 0;
    for (size_t i = 0; i < bnn_data.get_archi(nb_layers-1); i++) {
       if(activation[nb_layers-1][i]== 1)
       {
         predict = i;
         compt++;
       }
    }
    if(compt > 1){
      result =  false;
      std::cout<<"There is " << compt << " activated neurons on the output layer : "<<std::endl;
      std::cout<<"True neuron to be activated is " <<  example_label << std::endl;
        for (size_t i = 0; i < bnn_data.get_archi(nb_layers-1); i++)
          if(activation[nb_layers-1][i]== 1)
            std::cout<<"Neurone " << i << " at the last layer is activated"<<std::endl;
    }
    else{
      if(compt == 0){
        std::cout<<"There is no activated neuron on the output layer"<<std::endl;
        result =  false;
      }
    }

    if(predict != example_label){
      std::cout<<"The output label does not correspond to the expected one"<<std::endl;
      std::cout<<"True neuron to be activated is " <<  example_label << std::endl;
      std::cout<<"Activated neuron on the output layer is" << predict <<std::endl;
      result =  false;
    }
    return result;
  }


  /* predict method
  This function tests if the output of the network with the weights returned
  by the solver corresponds the label of the input
  Parameters :
   - index_example : index of the input example
   - test_set : boolean that indicates in which set to take the input example (default = test set)
   - verification_mode : boolean that indicates if verification logs have to be printed (default = false)
  Output : boolean -> true if the input is well classified and false either
  */
  bool predict(const int &index_example, const int &test_set = true, const bool &verification_mode = false){
    if(test_set){
      example_label = (int)bnn_data.get_dataset().test_labels[index_example];
      std::vector<uint8_t> temp = bnn_data.get_dataset().test_images[index_example];
      int size = temp.size();
      for (size_t i = 0; i < size; i++) {
        activation[0][i] = (int)temp[i];
      }
    }
    else{
      example_label = (int)bnn_data.get_dataset().training_labels[index_example];
      std::vector<uint8_t> temp = bnn_data.get_dataset().training_images[index_example];
      int size = temp.size();
      for (size_t i = 0; i < size; i++) {
        activation[0][i] = (int)temp[i];
      }
    }

    bool result = true ;
    for (size_t l = 1; l < nb_layers; l++) {
      for (size_t i = 0; i < bnn_data.get_archi(l-1); i++) {
        for (size_t j = 0; j < bnn_data.get_archi(l); j++) {
          preactivation[l][j] += activation[l-1][i] * weights[l-1][i][j];
        }
        for (size_t j = 0; j < bnn_data.get_archi(l); j++) {
          activation[l][j] = activation_function(preactivation[l][j]);
        }
      }
    }
    int predict = 0, compt = 0;;
    for (size_t i = 0; i < bnn_data.get_archi(nb_layers-1); i++) {
       if(activation[nb_layers-1][i]== 1)
       {
         predict = i;
         compt++;
       }
    }

    if (verification_mode) {
      if(compt > 1){
        result =  false;
        std::cout<<"There is " << compt << " activated neurons on the output layer : "<<std::endl;
        std::cout<<"True neuron to be activated is " <<  example_label << std::endl;
          for (size_t i = 0; i < bnn_data.get_archi(nb_layers-1); i++){
            if(activation[nb_layers-1][i]== 1)
              std::cout<<"Neuron " << i << " at the last layer is activated"<<std::endl;
          }
      }
      else{
        if(compt == 0){
          std::cout<<"There is no activated neuron on the output layer"<<std::endl;
          result =  false;
        }
      }

      if(predict != example_label){
        std::cout<<"The output label does not correspond to the expected one"<<std::endl;
        std::cout<<"True neuron to be activated is " <<  example_label << std::endl;
        std::cout<<"Activated neuron on the output layer is" << predict <<std::endl;
        result =  false;
      }
    }
    else {
      if (compt > 1 || compt == 0 || predict != example_label) {
        result = false;
      }
    }
    return result;
  }

  /* check_activation_preactivation method
  This function tests if the preactivation and activation values returned by the solver
  are the same as the one calculated by the predict function
  Parameters : None
  Output : boolean -> true if the all values are the same
  */
  bool check_activation_preactivation(){
    bool result = true;
    for (size_t l = 1; l < nb_layers; l++) {
      for (size_t j = 0; j < bnn_data.get_archi(l); j++) {
        if (preactivation[l][j] != solver_preactivation[l-1][j]) {
          result = false;
          std::cout << "The value of the preactivation for neuron "<<j<<" on layer "<<l<<" is incorrect." << '\n';
          std::cout << "Value is " <<solver_preactivation[l-1][j]<<" instead of "<<preactivation[l][j]<< '\n';
        }
      }
    }
    for (size_t l = 0; l < nb_layers; l++) {
      for (size_t j = 0; j < bnn_data.get_archi(l); j++) {
        if (activation[l][j] != solver_activation[l][j]) {
          result = false;
          std::cout << "The value of the activation for neuron "<<j<<" on layer "<<l<<" is incorrect." << '\n';
          std::cout << "Value is " <<solver_activation[l][j]<<" instead of "<<activation[l][j]<< '\n';
        }
      }
    }
    return result;
  }

  /* run_solution method
  This function runs all the methods above
  Parameters :
    - check_act_preact : boolean that indicates if the activation and preactivation values have to be checked
  Output : boolean -> true if the methods both return true
  */
  bool run_solution(const bool &check_act_preact){
    bool pred;
    bool act_preact = true;
    init(false);
    pred = predict();
    if (check_act_preact) {
      act_preact = check_activation_preactivation();
    }

    if (pred && act_preact) {
      std::cout << "OK" << '\n';
    }
    return (pred && act_preact);
  }



  /* run_solution method
  This function runs all the methods above
  Parameters :
    - check_act_preact : boolean that indicates if the activation and preactivation values have to be checked
    - index_example, test_set and verification_mode : arguments for the predict method
  Output : boolean -> true if the methods both return true
  */
  bool run_solution(const bool &check_act_preact, const int &index_example, const int &test_set, const bool &verification_mode){
    bool pred;
    bool act_preact = true;
    init(true);
    pred = predict(index_example, test_set, verification_mode);
    if (check_act_preact) {
      act_preact = check_activation_preactivation();
    }
    if (pred && act_preact && verification_mode) {
      std::cout << "OK" << '\n';
    }
    return (pred && act_preact);
  }


};





#endif
