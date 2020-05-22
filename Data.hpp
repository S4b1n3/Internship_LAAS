#ifndef DEF_DATA
#define DEF_DATA


#include "mnist-master/include/mnist/mnist_reader.hpp"
#include <cstdio>
#include <iostream>
#include <vector>

#define MNIST_DATA_LOCATION "/home/sabine/Documents/Seafile/Stage LAAS/or-tools_Ubuntu-18.04-64bit_v7.5.7466/tests/mnist-master"


/*  Class Data used to save the informations that are necessary for an instance of the problem
Attributs:
- nb_layers : number of layers of the BNN
- architecture : vector that contains the number of neurons on each layer
- dataset : MNIST training and testing sets whith their labels
*/
class Data{
private :
  int nb_layers;
  int nb_weights;
  std::vector<int> architecture;
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

public:

  /*
  Constructor of the class Data
  Argument :
  - a vector representing the architecture of a BNN
    The lenght of this vector is the number of layers of the BNN
  */
  explicit Data(const std::vector<int> &archi);


  /*
  Simple functions used to print the informations of BNN after its initialization
  */
  void print_archi();

  void print_dataset();


  /* Getters  */

  //returns the attribute nb_layers of the class Data
  int get_layers() const;

  //returns the whole architecture of the class Data
  std::vector<int> get_archi() const;

  //returns the number of neurons on the layer given as a parameter
  int get_archi(const int &layer) const;

  //returns the whole dataset of the class Data
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> get_dataset();

  //returns the number of weights in the whole network
  int get_nb_weigths() const;

};

#endif
