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

/*
 * Status codes :
 * 0 : UNKNOWN
 * 1 : MODEL_INVALID
 * 2 : FEASIBLE
 * 3 : INFEASIBLE
 * 4 : OPTIMAL
 */
std::vector<std::string> STATUS = {"UNKNOWN", "FEASIBLE", "INFEASIBLE", "OPTIMAL"};
//index of the experiment
int num_expe;

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


//Two methods to print vector
//Mostly used for tests and checking
void print_vector(const std::vector<std::string>& vecteur){
    for (const auto& i : vecteur)
        std::cout << i << std::endl;
}

void print_vector(const std::vector<int>& vecteur){
    for (const auto& i : vecteur)
        std::cout << i << std::endl;
}


/* Class Parser used to recover the satistics of the solver, wrote in different files
Attributs :
- input file : name of the file (output of the solver, input of the parser)
- nb_examples : number of examples as input of the model
Statistics of the solver :
- status : array that contains the number of status for each codes
- run time : mean of CPU time
- memory : mean memory usage
- nb_variables : mean of number of variables
- nb_constraints : mean of number of constraints
- objective_value : mean of the objective value when a solution has been returned
- nb_branches : mean of the number of branches in the search tree
- test_accuracy : accuracy on the testing set
- train_accuracy : accuracy on the testing set
*/
class Parser{
private:
    std::ifstream input_file;
    int nb_examples;
    std::string file_name;

    std::vector<int> status = {0, 0, 0, 0};
    int run_time = 0;
    std::uint64_t memory = 0;
    int nb_variables = 0;
    int nb_constraints = 0;
    int objective_value = 0;
    int nb_branches = 0;
    int nb_conflicts = 0;
    int test_accuracy = 0;
    int train_accuracy = 0;
    int test_accuracy_max = 0;
    int train_accuracy_max = 0;
    std::string input;

public:

    /* Constructor of the class Parser
    Arguments :
    - _path : name of the input file
    From the path, the method gets the number of examples
    */
    Parser(const std::string& _path):input_file(_path), file_name(_path){
        size_t index;
        index = _path.find_last_of("/");
        std::string filename = _path.substr(index+1);
        input = filename.substr(0, filename.size()-5);
        nb_examples = 100;

    }

    /* Getters */

    //returns the number of examples
    int get_nb_examples() const {
        return nb_examples;
    }

    //returns the number of tests with the status of index index_status
    int get_status(const int &index_status) const{
        return status[index_status];
    }

    //returns the full array containing all the status of each test
    std::vector<int> get_status() const {
        return status;
    }

    //returns the mean of CPU time
    int get_run_time() const{
        return run_time;
    }

    //returns the mean memory usage
    std::uint64_t get_memory() const{
      return memory;
    }

    //returns the mean of the number of variables
    int get_nb_variables() const{
        return nb_variables;
    }

    //returns the mean of the number of constraints
    int get_nb_constraints() const{
        return nb_constraints;
    }

    //returns the mean of the objective value when a solution is returned
    int get_objective_value() const{
        return objective_value;
    }

    //returns the mean of the number of branches
    int get_nb_branches() const{
        return nb_branches;
    }

    //returns the mean of the number of conflicts
    int get_nb_conflicts() const{
        return nb_conflicts;
    }

    //returns the mean of the accuray on the testing set
    int get_test_accuracy() const{
      return test_accuracy;
    }

    //returns the mean of the accuray on the training set
    int get_train_accuracy() const{
      return train_accuracy;
    }

    //returns the mean of the accuray on the testing set with all good metric method
    int get_test_accuracy_max() const{
      return test_accuracy_max;
    }

    //returns the mean of the accuray on the training set with all good metric method
    int get_train_accuracy_max() const{
      return train_accuracy_max;
    }

    std::string get_input() const{
      return input;
    }

    /* read_file method
      This method reads the file line by line, recover some statistics and
      calculate the mean (or the proportion) of each statistic for the last experiment
      Arguments : None
      Output : None
    */
    void read_file(){
        //copy of the global variable num_expe
        int expe_temp = num_expe;
        //variable that counts the number of runs
        int count = 0;
        //variable that counts the number of runs with status OPTIMAL or FEASIBLE
        int count_objective = 0;
	      int count_accuracy = 0;
        if(input_file){
            std::string line;
            while (std::getline(input_file, line)){
                if (expe_temp == 1){                                          //if we are reading the logs of the last experiment
		                int status_temp;

                    if(line.substr(0,12) == "d CHECKING 0"){
                      std::cout << " c Error checking from file "<<file_name << '\n';
                    }
                    if (line.substr(0, 22) == "d TRAIN_ACCURACY_TIME ") {
                      if (std::stoi(line.substr(22)) > 5000) {
                        std::cout << " c Accuracy computing on training set is more than 5000 seconds on file : "<< file_name << '\n';
                      }
                    }

                    if (line.substr(0, 21) == "d TEST_ACCURACY_TIME ") {
                      if (std::stoi(line.substr(21)) > 800) {
                        std::cout << " c Accuracy computing on testing set is more than 800 seconds on file : "<< file_name << '\n';
                      }
                    }

                    if(line.substr(0, 11) == "d RUN_TIME "){
                        run_time += std::stoi(line.substr(11));
                        count +=1;
                    }
                    if (line.substr(0, 9) == "d MEMORY ") {
                      memory += std::stoll(line.substr(9));
                    }
                    if(line.substr(0, 9) == "d STATUS "){
                        status_temp = std::stoi(line.substr(9));
                        switch (status_temp) {
                            case 0:
                                status[0] +=1;
                                break;
                            case 1:
                                std::cout << " c Error ! Model is invalid" << std::endl;
                                break;
                            default:
                                status[status_temp-1] += 1;
                                break;
                        }
                    }
                    if (line.substr(0, 12) == "d OBJECTIVE "){
		                    if(status_temp == 2 || status_temp == 4){
                            objective_value += std::stoi(line.substr(12));
                            count_objective += 1;
			                  }
                    }
                    if (line.substr(0, 11) == "d BRANCHES "){
                        nb_branches += std::stoi(line.substr(11));
                    }
                    if (line.substr(0, 12) == "d CONFLICTS "){
                        nb_conflicts += std::stoi(line.substr(12));
                    }
                    if(line.substr(0, 12) == "d VARIABLES "){
                        nb_variables += std::stoi(line.substr(12));
                    }
                    if(line.substr(0,14) == "d CONSTRAINTS "){
                        nb_constraints += std::stoi(line.substr(14));
                    }
                    if (line.substr(0, 23) == "d TEST_STRONG_ACCURACY ") {
		                    if(status_temp == 2 || status_temp == 4){
                    	     test_accuracy += std::stoi(line.substr(23));
			                     count_accuracy += 1;
			                  }
		                }
                    if (line.substr(0, 24) == "d TRAIN_STRONG_ACCURACY ") {
		                    if(status_temp == 2 || status_temp == 4){
                      	   train_accuracy += std::stoi(line.substr(24));
			                  }
                    }
                    if (line.substr(0, 21) == "d TEST_WEAK_ACCURACY ") {
		                    if(status_temp == 2 || status_temp == 4){
                    	     test_accuracy_max += std::stoi(line.substr(21));
			                  }
		                }
                    if (line.substr(0, 22) == "d TRAIN_WEAK_ACCURACY ") {
		                    if(status_temp == 2 || status_temp == 4){
                      	   train_accuracy_max += std::stoi(line.substr(22));
			                  }
                    }
                } else {
                    if (line.substr(0, 10) == "----------")
                        expe_temp--;
                }

            }
        } else
            std::cout << " c Error oppening file" << std::endl;
        if(count != 0){
          run_time = std::round(run_time/count);
          memory = memory/count;
          nb_variables /= count;
          nb_constraints /= count;
          nb_branches /= count;
          nb_conflicts /= count;
        } else {
          std::cout << " c No experiment written in output files" << '\n';
        }
        if (count_objective != 0)
            objective_value /= count_objective;

      	if(count_accuracy != 0){
      	    train_accuracy /= count_accuracy;
      	    test_accuracy /= count_accuracy;
            train_accuracy_max /= count_accuracy;
      	    test_accuracy_max /= count_accuracy;
      	}

        for (int i = 0; i < STATUS.size(); ++i) {
            status[i] = std::round(status[i]*100.0/count);
        }
    }

};


/* Class Parser_Container used to browse the tree of result files
and parse each file of the given folders
Attributs :
- parsers : container of Parser (the first dimension of the array corresponds
the folders and the second dimension is the files contained by the folder)
- path : path of the folder
- folders : array containing the name of the subfolders
- files : array containing the name of the files in each folder
- archi : architectures of the model (each subfolder contains the logs with a different architecture)
*/
class Parser_Container{
private:
    std::vector<std::vector<Parser*>> parsers;
    std::string path;
    std::vector<std::string> folders;
    std::vector<std::vector<std::string>> files;
    std::vector<std::vector<int>> archi;
    std::vector<std::string> inputs;

public:


    /* Constructor of the class Parser_Container
    This method reads the folder, recovers the architectures tested, reorders the folders,
    reads the subfolders and reorders the files inside each subfolder
    Arguments :
    - _path : path of the folder
    */
    explicit Parser_Container(std::string _path):path(std::move(_path)){

        folders = read_folder(path);
        archi.resize(folders.size());

        for (int i = 0; i < folders.size(); ++i) {
            if (folders[i].size() < 8){
              archi[i].push_back(0);
              break;
            }
            else
              get_archi(folders[i], i);
        }
        reorder_folders(archi);
        read_subfolders();
        reorder_files();
    }


    /* Getters */

    //returns all parsers of each file of the whole folder
    std::vector<std::vector<Parser*>> get_parsers(){
        return parsers;
    }

    //returns the parsers of each file of the subfolder i
    std::vector<Parser*> get_parsers(const int &i){
        return parsers[i];
    }

    //returns the parser of file j inside subfolder i
    Parser* get_parsers (const int &i, const int &j){
        return parsers[i][j];
    }

    //returns the architectures tested during the experiments
    std::vector<std::vector<int>> get_archi(){
        return archi;
    }

    //returns the names of the subfolders
    std::vector<std::string> get_folders(){
        return folders;
    }

    //returns the names of the files inside each subfolder
    std::vector<std::vector<std::string>> get_files(){
        return files;
    }

    //returns the name of the files inside subfolder i
    std::vector<std::string> get_files(const int &i){
        return files[i];
    }

    //returns the name of file j inside subfolder i
    std::string get_files(const int &i, const int &j){
        return files[i][j];
    }

    void get_archi(const std::string &folder_name, const int &index){
        std::string folder_temp = folder_name.substr(8);
        std::string temp;
        for (char j : folder_temp) {
            if(j == '_'){
                archi[index].push_back(std::stoi(temp));
                temp = "";
            } else
                temp += j;
        }
        archi[index].push_back(std::stoi(temp));
    }

    void get_strategy(const std::string &folder_name){
      std::string folder_temp = folder_name.substr(7);
      inputs.push_back(folder_temp);
    }

    /* read_folder method
    This method browses a folder and gets the name
    of the folders or the files that is contains
    Arguments :
    - _path : path of the folder to read
    Output : a vector containg the name of the subfolders or the files inside _path
    */
    static std::vector<std::string> read_folder(const std::string& _path) {
        std::vector<std::string> vector;
        DIR *dir;
        dirent *pdir;
        dir = opendir(_path.c_str());
        pdir = readdir(dir);
        while (pdir) {
            if (strcmp(pdir->d_name, ".") != 0 && strcmp(pdir->d_name, "..") != 0 && !strstr(pdir->d_name,".tex"))
                vector.emplace_back(pdir->d_name);
            pdir = readdir(dir);
        }
        return vector;
    }

    /* read_subfolders method
    For each subfolder found before, this method recovers the name of the files contained
    Arguments : None
    Output ; None
    */
    void read_subfolders(){
        files.resize(folders.size());
        for (int i = 0; i < folders.size(); ++i) {
          std::cout << path+"/"+folders[i] << '\n';
            files[i] = read_folder(path+"/"+folders[i]);
        }
    }

    /* reorder_folders method
    This method reorders the folders depending on the architectures tested
    Arguments : None
    Output : None
    */
    void reorder_folders(std::vector<std::vector<int>> vector){
        std::sort (vector.begin(), vector.end());
        std::vector<std::string> subfolder(vector.size());
        std::vector<std::string> temp_folders(folders.size());
        for (int i = 0; i < vector.size(); ++i) {
            for (int j = 0; j < vector[i].size(); ++j) {
                subfolder[i].append("_"+std::to_string(vector[i][j]));
            }
        }
        for (int i = 0; i < subfolder.size(); ++i) {
            for (auto & folder : folders) {
                if (folder.substr(7) == subfolder[i])
                    temp_folders[i] = folder;
            }
        }
        folders = temp_folders;
    }

    void reorder_folders(std::vector<std::string> vector){
        std::sort (vector.begin(), vector.end());
        std::vector<std::string> subfolder(vector.size());
        std::vector<std::string> temp_folders(folders.size());
        for (int i = 0; i < vector.size(); ++i) {
            subfolder[i].append(vector[i]);
        }
        for (int i = 0; i < subfolder.size(); ++i) {
            for (auto & folder : folders) {
                if (folder.substr(7) == subfolder[i])
                    temp_folders[i] = folder;
            }
        }
        folders = temp_folders;
    }



    /* reorder_files method
    This function reorders the files depending on the number of examples as input
    Arguments : None
    Output : None

    void reorder_files(){
        std::vector<std::vector<std::string>> temp;
        std::vector<std::vector<int>> nb_examples_temp;
        nb_examples_temp.resize(files.size());
        for (int j = 0; j < files.size(); ++j) {
            nb_examples_temp[j].resize(files[j].size());
            for (int i = 0; i < files[j].size(); ++i) {
                //std::cout << "stoi1 :"<< files[j][i].substr(7, files[j][i].size()-12) << '\n';
                nb_examples_temp[j][i] = std::stoi(files[j][i].substr(7, files[j][i].size()-5));
            }
        }
        for (int i = 0; i < files.size(); ++i) {
            std::sort(nb_examples_temp[i].begin(), nb_examples_temp[i].end());
        }
        temp.resize(nb_examples_temp.size());
        for (int i = 0; i < nb_examples_temp.size(); ++i) {
            for (int j = 0; j < nb_examples_temp[i].size(); ++j) {
                for (int k = 0; k < files[i].size(); ++k) {
                    //std::cout << "stoi2 :"<< files[i][k].substr(7, files[i][k].size()-12) << '\n';
                    if (std::stoi(files[i][k].substr(7, files[i][k].size()-5)) == nb_examples_temp[i][j]){
                        temp[i].push_back(files[i][k]);
                    }
                }
            }
        }
        files = temp;
    }*/

    void reorder_files(){
        std::vector<std::vector<std::string>> temp;
        std::vector<std::vector<std::string>> inputs_temp;
        inputs_temp.resize(files.size());
        for (int j = 0; j < files.size(); ++j) {
            inputs_temp[j].resize(files[j].size());
            for (int i = 0; i < files[j].size(); ++i) {
                //std::cout << "stoi1 :"<< files[j][i].substr(7, files[j][i].size()-12) << '\n';
                inputs_temp[j][i] = files[j][i].substr(0, files[j][i].size()-5);
            }
        }
        for (int i = 0; i < files.size(); ++i) {
            std::sort(inputs_temp[i].begin(), inputs_temp[i].end());
        }
        temp.resize(inputs_temp.size());
        for (int i = 0; i < inputs_temp.size(); ++i) {
            for (int j = 0; j < inputs_temp[i].size(); ++j) {
                for (int k = 0; k < files[i].size(); ++k) {
                    //std::cout << "stoi2 :"<< files[i][k].substr(7, files[i][k].size()-12) << '\n';
                    if (files[i][k].substr(0, files[i][k].size()-5) == inputs_temp[i][j]){
                        temp[i].push_back(files[i][k]);
                    }
                }
            }
        }
        files = temp;
    }


    /* create_parsers method
    For each file of the folder, this function creates a Parser from the class above
    Arguments : None
    Output : None
    */
    void create_parsers(){
        parsers.resize(files.size());
        for (int i = 0; i < files.size(); ++i){
            parsers[i].resize(files[i].size());
            for (int j = 0; j < files[i].size(); ++j) {
                auto *new_parser = new Parser(path+"/"+folders[i]+"/"+files[i][j]);
                new_parser->read_file();
                parsers[i][j] = new_parser;
            }
        }
    }

    /* end_expe method
    After reading all files, this function adds a "log" to indicate that
    we have reach the each of the logs of the current experiment index
    Arguments : None
    Output : None
    */
    void end_expe(){
        for (int i = 0; i < files.size(); ++i) {
            for (int j = 0; j < files[i].size(); ++j) {
                std::ofstream file(path+"/"+folders[i]+"/"+files[i][j], std::ios::app);
                if (file){
                    file << "----------"<< std::endl;
                } else
                    std::cout << " c Error oppening result file" << std::endl;
                file.close();
            }
        }
    }
};


/* Class Writer writes all the statistics recovered before on a .tex table
Attributs :
- result_file : ofstream used to write in the output file
- filename  : name of the output file
- nb_files : number of file inside each subfolders
- nb_folders : number of folders inside the main folder
- parsers : vector 2D that contains the parser of each file
- nb_examples : maximum number of input examples for each subfolder
- archi : vector 2D that contains the architectures
*/
class Writer{
private:
    std::ofstream result_file;
    std::string filename;
    std::vector<int> nb_files;
    const int nb_folders;
    std::vector<std::vector<Parser*>> parsers;
    std::vector<int> nb_examples;
    std::vector<std::vector<int>> archi;

public:

    /* Constructor of the class Writer
    Arguments :
    - path : path of the output file
    - _parsers : parser container
    */
    explicit Writer(const std::string &path, Parser_Container _parsers):
        nb_folders(_parsers.get_folders().size()), archi(_parsers.get_archi()){
        nb_files.resize(nb_folders);
        for (int i = 0; i < nb_folders; ++i) {
            nb_files[i] = (_parsers.get_files(i).size());
        }
        parsers.resize(nb_folders);
        for (int i = 0; i < nb_folders; ++i) {
            parsers[i].resize(nb_files[i]);
            for (int j = 0; j < nb_files[i]; ++j) {
                parsers[i][j] = _parsers.get_parsers(i, j);
            }
        }
        std::vector<int> max_examples(nb_folders,0);
        for (int i = 0; i < nb_folders; ++i) {
            for (int j = 0; j < nb_files[i]; ++j) {
                if (max_examples[i] < parsers[i][j]->get_nb_examples()) {
                    max_examples[i] = parsers[i][j]->get_nb_examples();
                }
            }
        }
        nb_examples = max_examples;



        size_t index;
        index = path.find_last_of("/");
        std::string folder_name = path.substr(index+1);
        filename = path + "/table-" +folder_name.substr(7) + "-E"+std::to_string(num_expe)+".tex";
    }

    /* Getters */

    //returns the output filename
    const std::string &get_filename() const {
        return filename;
    }


    /* print_header_table
    Arguments : None
    Output :
    - header : string containg the header commands in latex*/
    static std::string print_header_table() {
        std::string header(R"(\begin{table} [!ht] \centering \begin{tabular}{ ||c||c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c| } \hline \I & \V & \C & \B & \Cf & \UNK & \SAT  & \UNSAT & \OPT & \OBJ & \T & \M & \TSA & \TRA & \TSAW & \TRAW \\ \hline)");
        return header;
    }

    /* print_footer_table
    Arguments : None
    Output :
    - footer : string containg the footer commands in latex*/
    std::string print_footer_table(){
        size_t index;
        index = filename.find_last_of("/");
        std::string temp_ref = filename.substr(index+1);
        std::string footer("\\end{tabular} \\caption{Statistics with P"+temp_ref.substr(7, 1));
        if(temp_ref.size() == 17)
            footer.append(" and M2}");
        else
            footer.append(" and M1}");
        footer.append("\\label{tab:"+temp_ref.substr(6, temp_ref.size()-6-4)+"} \\end{table}");
        return footer;
    }




    /* print_multirow method
    Arguments :
    - index_folder : index of the folder
    Output :
    - multirow : string containing the multirow command in latex*/
    std::string print_multirow(const int &index_folder){
        std::string multirow("\\multicolumn{16}{|c|}{\\textbf{Architecture : 784,");
        for (int i : archi[index_folder]) {
            multirow.append(std::to_string(i)+",");
        }
        multirow.append("10}} \\\\ \\hline");
        return multirow;
    }

    /* print_parser method
    Giver the index the folder and the index of the file,
    this function writes all the informations about the file inside the table
    Arguments :
    - index_folder : index of the folder
    - index_file : index of the file
    */
    std::string print_parser(const int &index_folder, const int &index_file){
        Parser* temp = parsers[index_folder][index_file];
        std::string parser(temp->get_input()+" & "+std::to_string(temp->get_nb_variables())+" & "+
                        std::to_string(temp->get_nb_constraints())+" & ");
        if(temp->get_nb_branches() > 1000000){
          int temp_branches = (int)temp->get_nb_branches()/1000000;
          parser.append(std::to_string(temp_branches)+ "M & ");
        }
        else{
          if (temp->get_nb_branches() > 1000) {
            int temp_branches = (int)temp->get_nb_branches()/1000;
            parser.append(std::to_string(temp_branches)+ "k & ");
          }else{
            parser.append(std::to_string(temp->get_nb_branches())+ " & ");
          }
        }
        if(temp->get_nb_conflicts() > 1000000){
          int temp_conflicts = (int)temp->get_nb_conflicts()/1000000;
          parser.append(std::to_string(temp_conflicts)+ "M & ");
        }
        else{
          if (temp->get_nb_conflicts() > 1000) {
            int temp_conflicts = (int)temp->get_nb_conflicts()/1000;
            parser.append(std::to_string(temp_conflicts)+ "k & ");
          }else{
            parser.append(std::to_string(temp->get_nb_conflicts())+ " & ");
          }
        }
        for (int i = 0; i < STATUS.size(); ++i) {
            parser.append(std::to_string(temp->get_status(i))+" \\% & ");
        }
        if (temp->get_objective_value() != 0)
            parser.append(std::to_string(temp->get_objective_value())+" & ");
        else
            parser.append("0 & ");
        if (temp->get_run_time() < 1)
            parser.append("$<$1 & ");
        else
            parser.append(std::to_string(temp->get_run_time())+" & ");
        if(temp->get_memory() >= 1000000){
          int temp_mem = (int)temp->get_memory()/1000000;
          parser.append(std::to_string(temp_mem)+ "M & ");
        }
        else{
          if (temp->get_memory() >= 1000) {
            int temp_mem = (int)temp->get_memory()/1000;
            parser.append(std::to_string(temp_mem)+ "k & ");
          }else{
            parser.append(std::to_string(temp->get_memory())+ " & ");
          }
        }
        parser.append(std::to_string(temp->get_test_accuracy())+ "\\% & ");
        parser.append(std::to_string(temp->get_train_accuracy())+ "\\% & ");
        parser.append(std::to_string(temp->get_test_accuracy_max())+ "\\% & ");
        parser.append(std::to_string(temp->get_train_accuracy_max())+ "\\% \\\\ ");

        return parser;
    }



    /* run_parsing method
    This function runs all the methods above and write the results
    in the output file for each file inside each subfolder
    Arguments : None
    Output : None
    */
    void run_parsing() {
        result_file.open(filename, std::ios::out);
        if (result_file.bad())
            std::cout << " c Error oppening result file"<<std::endl;
        else{
            result_file << print_header_table() << std::endl;
            for (int i = 0; i < nb_folders; ++i) {
                result_file << print_multirow(i) << std::endl;
                for (int j = 0; j < nb_files[i]; ++j) {
                    result_file << print_parser(i, j) << std::endl;
                }
                result_file << "\\hline \\hline" << std::endl;
            }
            result_file << print_footer_table() << std::endl;
        }
    }
};


int main(int argc, char **argv) {
    num_expe = std::stoi(argv[2]);
    const std::string path_folder("/home/sabine/Documents/Seafile/Stage LAAS/or-tools_Ubuntu-18.04-64bit_v7.5.7466/BNN/results/"+std::string(argv[1])); ///home/smuzellec/or-tools_Ubuntu-18.04-64bit_v7.5.7466/
    //const std::string path_folder("/pfcalcul/tmp/smuzellec/or-tools_Ubuntu-18.04-64bit_v7.5.7466/rocknrun/bnn_cp_model/BNN/results/"+std::string(argv[1]));
    std::cout << path_folder <<std::endl << std::endl;
    Parser_Container first_test(path_folder);
    first_test.create_parsers();
    first_test.end_expe();

    Writer first_writer(path_folder, first_test);
    std::cout << first_writer.get_filename();
    first_writer.run_parsing();

    return 0;
}
