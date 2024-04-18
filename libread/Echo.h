// T. Grenier - Clanu 2019-2020 - INSA Lyon - GE

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef ECHO_PARSER_H
#define ECHO_PARSER_H

#include <string>
#include <vector>

# define ECHO_NB_LABEL 2


//! classe de lecture des données MNIST
class ECHO {
    std::vector<std::string> Dataset;
public:

//! Read the content of one directory, return the number of files present in the directory. This function shuffle the data.
    int ReadPath(std::string pathname);

//! Read images : start at offset, read only num_images
    void ReadInput(int offset, int num_images, float** inputs);
//! Read labels : start at offset, read only num_labels
    void ReadLabel(int offset, int num_labels, char **outputs);

//! static function
    static char convert_label( std::string const &s);
    static std::string convert_label( char s);


    static void PrintImage(float *im, int R, int C);
};

#endif
