#include <cstdio>
#include <iostream>
#include <vector>
#include <typeinfo>
#include <cmath>
#include <algorithm>

#include "Solution.hpp"



/* Constructor of the class Solution
Arguments :
- archi : architecture of the network
- weights : solution that has to be tested
- index_example : index of the input in the training set
*/
Solution::Solution(const std::vector<int> &archi, const std::vector<std::vector<std::vector<int>>> &_weights, const int &index_example): bnn_data(archi), weights(std::move(_weights)){
  nb_layers = bnn_data.get_layers();
  example_label = (int)bnn_data.get_dataset().training_labels[index_example];
  for (size_t i = 0; i < bnn_data.get_dataset().training_images[index_example].size(); i++) {
    example_images.push_back((int)bnn_data.get_dataset().training_images[index_example][i]);
  }
}

/* activation_function method
Given an preactivation x, returns 1 if the preactivation is positive and -1 either
*/
int Solution::activation_function(int x){
  if (x >= 0) return 1;
  else return -1;
}

/* init method
This function initialize the inputs and the output of each neuron of the network
Parameters : None
Output : None
*/
void Solution::init(){
  //inputs[i][j] is the value of ???

  inputs.resize(bnn_data.get_layers());
  for (size_t l = 0; l < bnn_data.get_layers(); l++) {
    inputs[l].resize(bnn_data.get_archi(l+1));
    for(size_t j = 0; j < bnn_data.get_archi(l+1); j++){
      inputs[l][j] = 0;
    }
  }

  outputs.resize(nb_layers);
   //outputs[i][j] is the value of ???
  for (size_t l = 0; l < nb_layers ; l++) {
    outputs[l].resize(bnn_data.get_archi(l));
    for (size_t i = 0; i < bnn_data.get_archi(l); i++) {
      if(l==0)
        outputs[l][i] = example_images[i];
      else
        outputs[l][i] = 0;
    }
  }
}


/* predict method
Parameters : None
Output : boolean -> true if the input is well classified and false either
*/
bool Solution::predict(){
  bool result = true ;
  init();
  for (size_t l = 1; l < nb_layers; l++) {
    for (size_t i = 0; i < bnn_data.get_archi(l-1); i++) {
      for (size_t j = 0; j < bnn_data.get_archi(l); j++) {
        inputs[l][j] += outputs[l-1][i] * weights[l-1][i][j];
      }
      for (size_t j = 0; j < bnn_data.get_archi(l); j++) {
        outputs[l][j] = activation_function(inputs[l][j]);
      }
    }
  }
  int predict = 0, compt = 0;;
  for (size_t i = 0; i < bnn_data.get_archi(nb_layers-1); i++) {
     if(outputs[nb_layers-1][i]== 1)
     {
       predict = i;
       compt++;
     }
  }
  if(compt > 1){
    result =  false;
    std::cout<<"There is" << compt << "activated activated neurons on the output layer : "<<std::endl;
    std::cout<<"True neuron to be activated is " <<  example_label << std::endl;
      for (size_t i = 0; i < bnn_data.get_archi(nb_layers-1); i++)
        if(outputs[nb_layers-1][i]== 1)
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
    std::cout<<"Activated neuron on the output layer  is" << predict <<std::endl;
    result =  false;
  }
  return result;
}
