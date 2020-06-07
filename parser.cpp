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

/*
 * Status codes :
 * 0 : UNKNOWN
 * 1 : MODEL_INVALID
 * 2 : FEASIBLE
 * 3 : INFEASIBLE
 * 4 : OPTIMAL
 */
std::vector<std::string> STATUS = {"UNKNOWN", "FEASIBLE", "INFEASIBLE", "OPTIMAL"};
int num_expe;

template <class Container>
void split(const std::string& str, Container& cont, char delim = ' ')
{
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delim)) {
        cont.push_back(token);
    }
}

void print_vector(const std::vector<std::string>& vecteur){
    for (const auto& i : vecteur)
        std::cout << i << std::endl;
}

void print_vector(const std::vector<int>& vecteur){
    for (const auto& i : vecteur)
        std::cout << i << std::endl;
}


class Parser{
private:
    std::ifstream input_file;
    int nb_examples;

    int run_time;
    std::vector<int> status = {0, 0, 0, 0};
    int nb_variables = 0;
    int nb_constraints = 0;
    int objective_value = 0;
    int nb_branches = 0;

public:

    Parser(const std::string& _path):input_file(_path){
        size_t index, index2;
        index = _path.find_last_of("/");
        std::string new_path = _path.substr(0, index);
        std::string filename = _path.substr(index+1);
        index2 = new_path.find_last_of("/");
        std::string folder_name = new_path.substr(index2+1);
        nb_examples = std::stoi(filename.substr(7, filename.size()-5));

    }

    int get_nb_examples() const {
        return nb_examples;
    }

    int get_status(const int &index_status) const{
        return status[index_status];
    }

    std::vector<int> get_status() const {
        return status;
    }

    int get_run_time() const{
        return run_time;
    }

    int get_nb_variables() const{
        return nb_variables;
    }

    int get_nb_constraints() const{
        return nb_constraints;
    }

    int get_objective_value() const{
        return objective_value;
    }

    int get_nb_branches() const{
        return nb_branches;
    }

    void read_file(){
        int expe_temp = num_expe;
        int count = 0;
        int count_objective = 0;
        if(input_file){
            std::string line;
            while (std::getline(input_file, line)){
                if (expe_temp == 1){
                    if(line.substr(0, 9) == "run time "){
                        run_time += std::stoi(line.substr(9));
                        count +=1;
                    }
                    if(line.substr(0, 7) == "status "){
                        int status_temp = std::stoi(line.substr(7));
                        switch (status_temp) {
                            case 0:
                                status[0] +=1;
                                break;
                            case 1:
                                std::cout << "Error ! Model is invalid" << std::endl;
                                break;
                            default:
                                status[status_temp-1] += 1;
                                break;
                        }
                    }
                    if (line.substr(0, 10) == "objective "){
                        if (std::stoi(line.substr(10)) != 0){
                            objective_value += std::stoi(line.substr(10));
                            count_objective += 1;
                        }
                    }
                    if (line.substr(0, 9) == "branches "){
                        nb_branches += std::stoi(line.substr(9));
                    }
                    if(line.substr(0, 12) == "#Variables: "){
                        std::string temp;
                        for (auto i : line.substr(12)) {
                            if (i == ' ')
                                break;
                            else
                                temp += i;
                        }
                        nb_variables += std::stoi(temp);
                    }
                    if(line.substr(0,2) == "#k"){
                        std::vector<std::string> temp;
                        split(line, temp, ' ');
                        nb_constraints += std::stoi(temp[1]);
                    }
                } else {
                    if (line == "----------")
                        expe_temp--;
                }

            }
        } else
            std::cout << "File is not open" << std::endl;
        run_time = std::round(run_time/count);
        nb_variables /= count;
        nb_constraints /= count;
        if (count_objective != 0)
            objective_value /= count_objective;
        nb_branches /= count;
        for (int i = 0; i < STATUS.size(); ++i) {
            status[i] = std::round(status[i]*100.0/count);
        }
    }

};

class Parser_Container{
private:
    std::vector<std::vector<Parser*>> parsers;
    std::string path;
    std::vector<std::string> folders;
    std::vector<std::vector<std::string>> files;
    std::vector<std::vector<int>> archi;

public:

    explicit Parser_Container(std::string _path):path(std::move(_path)){
        folders = read_folder(path);
        archi.resize(folders.size());
        std::string temp;
        for (int i = 0; i < folders.size(); ++i) {
            std::string folder_temp = folders[i].substr(8);
            for (char j : folder_temp) {
                if(j == '_'){
                    archi[i].push_back(std::stoi(temp));
                    temp = "";
                } else
                    temp += j;
            }
            archi[i].push_back(std::stoi(temp));
            temp = "";
        }
        reorder_folders();
        read_subfolders();
        reorder_files();
    }

    std::vector<std::vector<Parser*>> get_parsers(){
        return parsers;
    }

    std::vector<Parser*> get_parsers(const int &i){
        return parsers[i];
    }

    Parser* get_parsers (const int &i, const int &j){
        return parsers[i][j];
    }

    std::vector<std::vector<int>> get_archi(){
        return archi;
    }

    std::vector<std::string> get_folders(){
        return folders;
    }

    std::vector<std::vector<std::string>> get_files(){
        return files;
    }

    std::vector<std::string> get_files(const int &i){
        return files[i];
    }

    std::string get_files(const int &i, const int &j){
        return files[i][j];
    }

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

    void read_subfolders(){
        files.resize(folders.size());
        for (int i = 0; i < folders.size(); ++i) {
            files[i] = read_folder(path+"/"+folders[i]);
        }
    }

    void reorder_files(){
        std::vector<std::vector<std::string>> temp;
        std::vector<std::vector<int>> nb_examples_temp;
        nb_examples_temp.resize(files.size());
        for (int j = 0; j < files.size(); ++j) {
            nb_examples_temp[j].resize(files[j].size());
            for (int i = 0; i < files[j].size(); ++i) {
                nb_examples_temp[j][i] = std::stoi(files[j][i].substr(7, files[i].size()-5));
            }
        }
        for (int i = 0; i < files.size(); ++i) {
            std::sort(nb_examples_temp[i].begin(), nb_examples_temp[i].end());
        }
        temp.resize(nb_examples_temp.size());
        for (int i = 0; i < nb_examples_temp.size(); ++i) {
            for (int j = 0; j < nb_examples_temp[i].size(); ++j) {
                for (int k = 0; k < files[i].size(); ++k) {
                    if (std::stoi(files[i][k].substr(7, files[i][k].size()-5)) == nb_examples_temp[i][j]){
                        temp[i].push_back(files[i][k]);
                    }
                }
            }
        }
        files = temp;
    }

    void reorder_folders(){
        std::sort (archi.begin(), archi.end());
        std::vector<std::string> subfolder(archi.size());
        std::vector<std::string> temp_folders(folders.size());
        for (int i = 0; i < archi.size(); ++i) {
            for (int j = 0; j < archi[i].size(); ++j) {
                subfolder[i].append("_"+std::to_string(archi[i][j]));
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

    void end_expe(){
        for (int i = 0; i < files.size(); ++i) {
            for (int j = 0; j < files[i].size(); ++j) {
                std::ofstream file(path+"/"+folders[i]+"/"+files[i][j], std::ios::app);
                if (file){
                    file << "----------";
                } else
                    std::cout << "Error oppening result file" << std::endl;
                file.close();
            }
        }
    }
};


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

    const std::string &get_filename() const {
        return filename;
    }

    static std::string print_header_table() {
        std::string header(R"(\begin{table} [!ht] \centering \begin{tabular}{ ||c||c|c|c|c|c|c|c|c|c|c| } \hline Architecture & \N & \V & \C & \B & \UNK & \SAT  & \UNSAT & \OPT & \OBJ & \T  \\ \hline)");
        return header;
    }

    std::string print_foot_table(){
        size_t index;
        index = filename.find_last_of("/");
        std::string temp_ref = filename.substr(index+1);
        std::string foot("\\end{tabular} \\caption{Statistics with "+temp_ref.substr(6, temp_ref.size()-6-8)+" neurons}"
                            "\\label{tab:"+temp_ref.substr(6, temp_ref.size()-6-4)+"} \\end{table}");
        return foot;
    }

    std::string print_multirow(const int &index_folder){
        std::string multirow("\\multirow{"+std::to_string(nb_examples[index_folder])+"}{4em}{784,");
        for (int i : archi[index_folder]) {
            multirow.append(std::to_string(i)+",");
        }
        multirow.append("10}");
        return multirow;
    }

    std::string print_parser(const int &index_folder, const int &index_file){
        Parser* temp = parsers[index_folder][index_file];
        std::string parser("& "+std::to_string(temp->get_nb_examples())+" & "+std::to_string(temp->get_nb_variables())+" & "+
                        std::to_string(temp->get_nb_constraints())+" & "+std::to_string(temp->get_nb_branches())+" & ");
        for (int i = 0; i < STATUS.size(); ++i) {
            parser.append(std::to_string(temp->get_status(i))+" \\% & ");
        }
        if (temp->get_objective_value() != 0)
            parser.append(std::to_string(temp->get_objective_value())+" & ");
        else
            parser.append(" & ");
        if (temp->get_run_time() < 1)
            parser.append("$<$1 \\\\ ");
        else
            parser.append(std::to_string(temp->get_run_time())+" \\\\ ");
        return parser;
    }




    void run_parsing() {
        result_file.open(filename, std::ios::out);
        if (result_file.bad())
            std::cout << "Error oppening file"<<std::endl;
        else{
            result_file << print_header_table() << std::endl;
            for (int i = 0; i < nb_folders; ++i) {
                result_file << print_multirow(i) << std::endl;
                for (int j = 0; j < nb_files[i]; ++j) {
                    result_file << print_parser(i, j) << std::endl;
                }
                result_file << "\\hline \\hline" << std::endl;
            }
            result_file << print_foot_table() << std::endl;
        }
    }
};


int main() {
    num_expe = 1;
    //const std::string path("/home/smuzellec/or-tools_Ubuntu-18.04-64bit_v7.5.7466/BNN/");
    const std::string path_file("/home/sabine/Documents/Seafile/Stage LAAS/or-tools_Ubuntu-18.04-64bit_v7.5.7466/BNN/");
    const std::string path_folder("/home/sabine/Documents/Seafile/Stage LAAS/or-tools_Ubuntu-18.04-64bit_v7.5.7466/BNN/results/results10N");
    Parser_Container first_test(path_folder);
    first_test.create_parsers();
    //first_test.end_expe();


    Writer first_writer(path_folder, first_test);
    std::cout << first_writer.get_filename();
    first_writer.run_parsing();

    return 0;
}
