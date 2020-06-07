
/*
 * data.h
 *
 *  Created on: Jun 7, 2020
 *      Author: msiala
 */



#ifndef EXAMPLES_CPP_DATA_H_
#define EXAMPLES_CPP_DATA_H_

#include "mnist-master/include/mnist/mnist_reader.hpp"

#define MNIST_DATA_LOCATION "mnist-master"
//#define MNIST_DATA_LOCATION "/home/smuzellec/or-tools_Ubuntu-18.04-64bit_v7.5.7466/BNN/mnist-master"


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
  explicit Data(const std::vector<int> &archi){
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
  void print_archi(){
    for (size_t i = 0; i < nb_layers; i++) {
      std::cout<<"Layer "<<i<<" : "<<architecture[i]<<" neurons"<<std::endl;
    }
  }

  void print_dataset(){
    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

  }

  /* Getters  */

  //returns the attribute nb_layers of the class Data
  int get_layers() const{
    return nb_layers;
  }

  //returns the whole architecture of the class Data
  std::vector<int> get_archi() const{
    return architecture;
  }

  //returns the number of neurons on the layer given as a parameter
  int get_archi(const int &layer) const{
    return architecture[layer];
  }

  //returns the whole dataset of the class Data
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> get_dataset(){
    return dataset;
  }

  //returns the number of weights in the whole network
  int get_nb_weigths() const{
    return nb_weights;
  }

};


#endif
