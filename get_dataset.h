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

void per_label(const int &nb_examples_per_label, const std::string &path) {
  Data *bnn_data = new Data();
  int nb_examples = 10*nb_examples_per_label;
  int compt_ex = 0;
  std::vector<int> occ(10);
  std::vector<int> ind;
  std::string output_path = path+"/per_label_"+std::to_string(nb_examples_per_label)+".data";
  std::ofstream file(output_path.c_str(), std::ios::out);
  while (compt_ex < nb_examples) {
    int index_rand = rand()%60000;
    auto it = std::find(std::begin(ind), std::end(ind), index_rand);
    if (it == ind.end()) {
      ind.push_back(index_rand);
      int label = (int)bnn_data->get_dataset().training_labels[index_rand];
      if(occ[label] < nb_examples_per_label){
        file << "LABEL " << label << std::endl;
        file << "IMAGE ";
        for (size_t i = 0; i < 784; i++) {
          file << (int)bnn_data->get_dataset().training_images[index_rand][i] << " ";
        }
        file<<std::endl;
        compt_ex++;
        occ[label]++;
      }
    }
  }

  file.close();
  delete bnn_data;
}

void random(const int &nb_examples, const std::string &path) {
  Data *bnn_data = new Data();
  int index_rand = rand()%(60000-nb_examples);
  std::string output_path = path+"/random_"+std::to_string(nb_examples)+".data";
  std::ofstream file(output_path.c_str(), std::ios::out);
  for (size_t i = 0; i < nb_examples; i++) {
    int label = (int)bnn_data->get_dataset().training_labels[index_rand+i];
    file << "LABEL " << label << std::endl;
    file << "IMAGE ";
    for (size_t j = 0; j < 784; j++) {
      file << (int)bnn_data->get_dataset().training_images[index_rand+i][j] << " ";
    }
    file<<std::endl;
  }
  file.close();
  delete bnn_data;
}

void correct(const std::string &output_file, const std::string &_input_file) {
  std::ifstream input_file(_input_file.c_str());

  std::vector<int> architecture;
  std::vector<std::vector<std::vector<int>>> weights;
  if(input_file){
    std::string line;

    while (std::getline(input_file, line)){
      if (line.substr(0, 6) == "ARCHI ") {
        std::string temp;
        for (size_t i = 6; i < line.size(); i++) {
          if (line[i] != ' ') {
            temp += line[i];
          }
          if (line[i] == ' ') {
            architecture.push_back(std::stoi(temp));
            temp = "";
          }
        }
      }
      if (line.substr(0, 8) == "WEIGHTS ") {
        int index_str = 8;
        weights.resize(architecture.size());
        for (size_t l = 1; l < architecture.size(); l++) {
          weights[l-1].resize(architecture[l-1]);
          for (size_t i = 0; i < architecture[l-1]; i++) {
            weights[l-1][i].resize(architecture[l]);
            for (size_t j = 0; j < architecture[l]; j++) {
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
    std::cout << "Error oppening input file" << '\n';
  }

  std::vector<int> correct_examples;
  Data *bnn_data = new Data (architecture);
  Evaluation test(weights, bnn_data);
  correct_examples = test.get_correct_examples(false);

  std::ofstream file(output_file.c_str(), std::ios::out);

  for(const auto &i : correct_examples){
    int label = (int)bnn_data->get_dataset().training_labels[i];
    file << "LABEL " << label << std::endl;
    file << "IMAGE ";
    for (size_t j = 0; j < 784; j++) {
      file << (int)bnn_data->get_dataset().training_images[i][j] << " ";
    }
    file<<std::endl;
  }
  file.close();
  delete bnn_data;
}


#endif
