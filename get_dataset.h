#ifndef EXAMPLES_CPP_GET_DATASET_H_
#define EXAMPLES_CPP_GET_DATASET_H_

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <cstdlib>
#include <dirent.h>
#include <cstring>
#include <sstream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <cinttypes>

#include "data.h"
#include "evaluation.h"

//method used to split a string given a delimiter and put the result in a container
template <class Container>
void split(const std::string& str, Container& cont, char delim = ' ')
{
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delim)) {
        cont.push_back(token);
    }
}

int rand_a_b(int a, int b){
	return rand()%((b-a)+1)+a;
}

bool fexists(const std::string& filename) {
  std::ifstream ifile(filename.c_str());
  return (bool)ifile;
}

void per_label(const int &nb_examples_per_label, const std::string &path, const int &index_file) {
  Data bnn_data;
  int nb_examples = 10*nb_examples_per_label;
  int compt_ex = 0;
  std::vector<int> occ(10);
  std::vector<int> ind;
  std::vector<int> idx_examples;
  std::string output_path = path+"/per_label_"+std::to_string(nb_examples_per_label)+"_"+std::to_string(index_file)+".data";
  std::ofstream file(output_path.c_str(), std::ios::out);
  while (compt_ex < nb_examples) {
    int index_rand = rand()%60000;
    auto it = std::find(std::begin(ind), std::end(ind), index_rand);
    if (it == ind.end()) {
      ind.push_back(index_rand);
      int label = (int)bnn_data.get_dataset().training_labels[index_rand];
      if(occ[label] < nb_examples_per_label){
        idx_examples.push_back(index_rand);
        compt_ex++;
        occ[label]++;
      }
    }
  }
  file << "INDEXES ";
  for (size_t i = 0; i < idx_examples.size(); i++) {
    file << idx_examples[i] << " ";
  }
  file << std::endl;
  file.close();
}

void random(const int &nb_examples, const std::string &path, const int &index_file) {
  Data bnn_data;
  int index_rand = rand()%(60000-nb_examples);
  std::string output_path = path+"/random_"+std::to_string(nb_examples)+"_"+std::to_string(index_file)+".data";
  std::ofstream file(output_path.c_str(), std::ios::out);
  file << "INDEXES ";
  for (size_t i = 0; i < nb_examples; i++) {
    file << index_rand+i << " ";
  }
  file << std::endl;
  file.close();
}

void correct(const std::string &_output_file, const std::string &_input_file, const std::vector<int> &_architecture) {
  std::ifstream input_file(_input_file.c_str());
  std::vector<std::vector<std::vector<int>>> weights;
  std::ofstream file(_output_file.c_str(), std::ios::out);

  if(!fexists(_input_file)){
    std::cout << " c creating solution" << '\n';
    std::ofstream solution(_input_file.c_str(), std::ios::out);


    solution << "ARCHI ";
    for (size_t i = 0; i < _architecture.size(); i++) {
      solution << _architecture[i] << " ";
    }
    solution << std::endl;
    solution << "WEIGHTS ";

    int tmp = _architecture.size()-1;
  	weights.resize(tmp);
  	for (size_t i = 1; i < tmp+1; i++) {
  		int tmp2 = _architecture[i-1];
  		weights[i-1].resize(tmp2);
  		for (size_t j = 0; j < tmp2; j++) {
  			int tmp3 = _architecture[i];
  			weights[i-1][j].resize(tmp3);
  			for (size_t k = 0; k < tmp3; k++) {
  				weights[i-1][j][k] = rand_a_b(-1,1);
          solution << weights[i-1][j][k] << " ";
  			}
  		}
  	}
  }else {
    if(input_file){
      std::string line;

      while (std::getline(input_file, line)){
        if (line.substr(0, 8) == "WEIGHTS ") {
          int index_str = 8;
          weights.resize(_architecture.size());
          for (size_t l = 1; l < _architecture.size(); l++) {
            weights[l-1].resize(_architecture[l-1]);
            for (size_t i = 0; i < _architecture[l-1]; i++) {
              weights[l-1][i].resize(_architecture[l]);
              for (size_t j = 0; j < _architecture[l]; j++) {
                if (line[index_str] == '-') {
                  weights[l-1][i][j] = -1;
                  index_str += 3;
                }
                else {
                  weights[l-1][i][j] = line[index_str] - '0';
                  index_str += 2;
                }
              }
            }
          }
        }
      }
    } else {
      std::cout << "Error oppening input file : " << _input_file << '\n';
    }
  }


  std::vector<int> correct_examples;
  Data bnn_data(_architecture);
  Evaluation test(weights, bnn_data);
  correct_examples = test.get_correct_examples(false);


  file << "INDEXES ";
  for(const auto &i : correct_examples){
    file << i << " ";
  }
  file << std::endl;
  file.close();
  std::cout << " c nb correct examples : " << correct_examples.size() << '\n';
}


#endif
