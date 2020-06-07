
#include "data.h"
#include "solution.h"
#include "cp_minweight_model.h"

#include <string>
#include <vector>


int main(int argc, char **argv) {

  srand(time(NULL));

  for (int i = 0; i < argc; ++i)
    std::cout << argv[i] << " ";


  std::vector<int> archi_test;
  int nb_neurons = 0;

  archi_test.push_back(784);
  int nb_examples = std::stoi(argv[1]);
  for (int i = 2; i < argc; ++i) {
    archi_test.push_back(std::stoi(argv[i]));
    nb_neurons += std::stoi(argv[i]);
  }
  archi_test.push_back(10);

  std::string filename("BNN/results/results"+std::to_string(nb_neurons)+"N/results");
  for (size_t i = 2; i < argc; i++) {
    filename.append("_"+std::string(argv[i]));
  }

  filename.append("/results"+std::to_string(nb_examples)+".stat");

  std::cout << filename << std::endl;


  operations_research::sat::CPModel_MinWeight first_model(archi_test, nb_examples);

  std::cout<<std::endl<<std::endl;

  first_model.run(1200.0) ;

  first_model.print_statistics(filename) ;
  //first_model.print_solution_bis(first_model.get_response());
  //first_model.print_all_solutions() ;

  /*std::vector<std::vector <std::vector<int>>> solution = first_model.get_solution();
  Solution first_solution(archi_test, solution, 0);
  std::cout << "Verification de la solution : "<<first_solution.predict() << '\n';*/

  return EXIT_SUCCESS;
}
