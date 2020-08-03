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

#include "get_dataset.h"


int main(int argc, char **argv) {
    int seed = atoi(argv[3]);
    srand(seed);
    const std::string path_folder("dataset");

    int sampling = atoi(argv[1]);
    int nb_ex = atoi(argv[2]);
    int index = atoi(argv[3]);

    if (sampling == 1) {
      per_label(nb_ex, path_folder, index);
    }
    else {
      if (sampling == 2) {
        random(nb_ex, path_folder, index);
      }else{
        std::cout << "Please enter 1 to generate \"per_label\" sampling or 2 to generate \"pur random\" sampling" << '\n';
      }
    }

    return 0;
}
