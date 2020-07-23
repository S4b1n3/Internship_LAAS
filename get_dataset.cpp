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

Data bnn_data;



void per_label(const int &nb_examples_per_label, const std::string &path) {
  int nb_examples = 10*nb_examples_per_label;
  int compt_ex = 0;
  std::vector<int> occ(10);
  std::vector<int> ind;
  std::string output_path = path+"/per_label_"+std::to_string(nb_examples_per_label)+".txt";
  std::ofstream file(output_path.c_str(), std::ios::out);
  while (compt_ex < nb_examples) {
    int index_rand = rand()%60000;
    auto it = std::find(std::begin(ind), std::end(ind), index_rand);
    if (it == ind.end()) {
      ind.push_back(index_rand);
      int label = (int)bnn_data.get_dataset().training_labels[index_rand];
      if(occ[label] < nb_examples_per_label){
        file << "LABEL " << label << std::endl;
        file << "IMAGE ";
        for (size_t i = 0; i < 784; i++) {
          file << (int)bnn_data.get_dataset().training_images[index_rand][i] << " ";
        }
        file<<std::endl;
        compt_ex++;
        occ[label]++;
      }
    }
  }

  file.close();
}

void random(const int &nb_examples, const std::string &path) {
  int index_rand = rand()%(60000-nb_examples);
  std::string output_path = path+"/random_"+std::to_string(nb_examples)+".txt";
  std::ofstream file(output_path.c_str(), std::ios::out);
  for (size_t i = 0; i < nb_examples; i++) {
    int label = (int)bnn_data.get_dataset().training_labels[index_rand+i];
    file << "LABEL " << label << std::endl;
    file << "IMAGE ";
    for (size_t j = 0; j < 784; j++) {
      file << (int)bnn_data.get_dataset().training_images[index_rand+i][j] << " ";
    }
    file<<std::endl;
  }
  file.close();
}

int main(int argc, char **argv) {
    srand(time(NULL));
    //const std::string path_folder("/pfcalcul/tmp/smuzellec/or-tools_Ubuntu-18.04-64bit_v7.5.7466/rocknrun/bnn_cp_model/BNN/results/"+std::string(argv[1]));
    const std::string path_folder("/home/sabine/Documents/Seafile/Stage LAAS/or-tools_Ubuntu-18.04-64bit_v7.5.7466/BNN/"+std::string(argv[3]));
    std::cout << path_folder <<std::endl << std::endl;

    int sampling = atoi(argv[1]);
    int nb_ex = atoi(argv[2]);

    if (sampling == 1) {
      per_label(nb_ex, path_folder);
    }
    else {
      if (sampling == 2) {
        random(nb_ex, path_folder);
      }else{
        std::cout << "Please enter 1 to generate \"per_label\" sampling or 2 to generate \"random\" sampling" << '\n';
      }
    }

    return 0;
}
