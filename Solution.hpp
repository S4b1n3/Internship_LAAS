#ifndef DEF_SOLUTION
#define DEF_SOLUTION

#include <cstdio>
#include <iostream>
#include <vector>
#include <typeinfo>
#include <cmath>
#include <algorithm>

#include "Data.hpp"

/* Class solution used to check that the solution returned by the solver is correct
Attributs :
- bnn_data : data of the network
- weights : values of the weights that have to be tested
- nb_layers : number of layer in the network
- inputs : input values for each neuron
- output : output values for each neuron
- example_images : inputs of the network
- example_label : labels of the inputs (the output must be equal to the label)
*/
class Solution {

private:
  Data bnn_data;
  std::vector<std::vector<std::vector<int>>> weights;
  int nb_layers;
  std::vector<std::vector<int>> inputs;
  std::vector<std::vector<int>> outputs;
  std::vector<int> example_images;
  int example_label;

public:

  /* Constructor of the class Solution
  Arguments :
  - archi : architecture of the network
  - weights : solution that has to be tested
  - index_example : index of the input in the training set
  */
  Solution(const std::vector<int> &archi, const std::vector<std::vector<std::vector<int>>> &_weights, const int &index_example);

  /* activation_function method
  Given an preactivation x, returns 1 if the preactivation is positive and -1 either
  */
  static int activation_function(int x);

  /* init method
  This function initialize the inputs and the output of each neuron of the network
  Parameters : None
  Output : None
  */
  void init();


  /* predict method
  Parameters : None
  Output : boolean -> true if the input is well classified and false either
  */
  bool predict();

};

#endif
