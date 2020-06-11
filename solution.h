
/*
 * solution.h
 *
 *  Created on: Jun 7, 2020
 *      Author: msiala
 */



#ifndef EXAMPLES_CPP_SOLUTION_H_
#define EXAMPLES_CPP_SOLUTION_H_



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
  std::vector<int> example_images;
  int example_label;

public:

  /* Constructor of the class Solution
  Arguments :
  - archi : architecture of the network
  - _weights : solution that has to be tested
  - _preactivation : preactivation values for each neuron returned by the solver
  - _activation : activation values for each neuron returned by the solver
  - index_example : index of the input in the training set
  */
  Solution(const std::vector<int> &archi, std::vector<std::vector<std::vector<int>>> _weights, std::vector<std::vector<int>> _activation, std::vector<std::vector<int>> _preactivation, const int &index_example):
  bnn_data(archi), weights(std::move(_weights)), solver_activation(std::move(_activation)), solver_preactivation(std::move(_preactivation)){
    nb_layers = bnn_data.get_layers();
    example_label = (int)bnn_data.get_dataset().training_labels[index_example];
    for (size_t i = 0; i < bnn_data.get_dataset().training_images[index_example].size(); i++) {
      example_images.push_back((int)bnn_data.get_dataset().training_images[index_example][i]);
    }
  }

  /* activation_function method
  Given an preactivation x, returns 1 if the preactivation is positive and -1 either
  */
  static int activation_function(int x){
    if (x >= 0) return 1;
    else return -1;
  }

  /* init method
  This function initialize the preactivation and the activation of each neuron of the network
  Parameters : None
  Output : None
  */
  void init(){
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
        if(l==0)
          activation[l][i] = example_images[i];
        else
          activation[l][i] = 0;
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
    int predict = 0, compt = 0;;
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
  Parameters : None
  Output : boolean -> true if the methods both return true
  */
  bool run_solution(){
    bool pred, act_preact;
    init();
    pred = predict();
    act_preact = check_activation_preactivation();
    if (pred && act_preact) {
      std::cout << "OK" << '\n';
    }
    return (pred && act_preact);
  }


};





#endif