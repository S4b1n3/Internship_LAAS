#include "Data.hpp"
#include "mnist-master/include/mnist/mnist_reader.hpp"

#include <cstdio>
#include <iostream>
#include <vector>

#define MNIST_DATA_LOCATION "/home/sabine/Documents/Seafile/Stage LAAS/or-tools_Ubuntu-18.04-64bit_v7.5.7466/tests/mnist-master"


/*
Constructor of the class Data
Argument :
- a vector representing the architecture of a BNN
  The lenght of this vector is the number of layers of the BNN
*/
Data::Data(const std::vector<int> &archi){
  nb_layers = archi.size();
  architecture = archi;
  nb_weights = 0;
  for (size_t l = 1; l < nb_layers; ++l) {
      nb_weights += architecture[l]*architecture[l-1];
  }
}

/*
Simple functions used to print the informations of BNN after its initialization
*/
void Data::print_archi(){
  for (size_t i = 0; i < nb_layers; i++) {
    std::cout<<"Layer "<<i<<" : "<<architecture[i]<<" neurons"<<std::endl;
  }
}

void Data::print_dataset(){
  std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
  std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
  std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
  std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

}

/* Getters  */

//returns the attribute nb_layers of the class Data
int Data::get_layers() const{
  return nb_layers;
}

//returns the whole architecture of the class Data
std::vector<int> Data::get_archi() const{
  return architecture;
}

//returns the number of neurons on the layer given as a parameter
int Data::get_archi(const int &layer) const{
  return architecture[layer];
}

//returns the whole dataset of the class Data
mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> Data::get_dataset(){
  return dataset;
}

//returns the number of weights in the whole network
int Data::get_nb_weigths() const{
  return nb_weights;
}
