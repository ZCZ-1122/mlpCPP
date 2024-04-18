///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef CT2D_PARSER_H
#define CT2D_PARSER_H

#include <string>
//#include <tuple>

//! classe de lecture des données MNIST
class CT2D {
public:


//! lecture des données 'image' (les entrées du réseau)
    void ReadInput(std::string pathname, int num_images, float** inputs);
//! lecture des étiquettes (les sorties du réseau, pour l'entrainement)
    void ReadLabel(std::string pathname, int num_labels, float** outputs);


    //                              data           w     h    Label   path
//    using ImageType = std::tuple<unsigned char *, int , int , char, std::string > ;

    static char convert_label( std::string const &s);
};



#endif
