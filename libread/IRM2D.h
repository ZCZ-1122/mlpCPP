// T. Grenier - Clanu 2021-2022 - INSA Lyon - GE

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef IRM2D_PARSER_H
#define IRM2D_PARSER_H

#include <string>
#include <vector>

#define IRM2D_NB_LABEL 9


//! classe de lecture des données IRM
class IRM2D {  
public:
    std::vector<std::string> Dataset;

//! Read the content of one directory, return the number of files present in the directory. This function shuffle the data.
    int ReadPath(std::string pathname);

//! Read images : start at offset, read only num_images
    void ReadInput(int offset, int num_images, float** inputs);
//! Read labels : start at offset, read only num_labels
    void ReadLabel(int offset, int num_labels, char** outputs);

//! static functions for label convertion from a string to a int
    static char convert_label( std::string const &s);
//! static functions for label convertion from a int to a string
    static std::string convert_label(char s);

    //! static function for image print (ASCII word art)
    static void PrintImage(float *im, int R, int C);

//! Print content of DataSet (filenames of images)
//! \param offset : first entry to print
//! \param num_images : number of entries to print. values under 0 means "all"
//    void PrintDataSet(int offset=0, int num_images=-1) const;

//! Save an image (im) to a given file (\param filename), provide also the width (R) and the height (C) of the image.
    void SaveImage(const char *filename, float *im, int R, int C) const;
};

#endif
