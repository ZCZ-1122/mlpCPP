// T. Grenier - Clanu 2019-2020 - INSA Lyon - GE

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#include "Echo.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>    // std::random_shuffle

#include <filesystem>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "timing_functions.h"
#include "progress_bar.h"

namespace fs = std::filesystem;

int ECHO::ReadPath(std::string pathname)
{
    // Clear existing list of files
    Dataset.clear();

    std::cout << "Reading Directory : " << pathname << std::endl;
    fs::path data_path(pathname);

// Compute number of files in the directory
    tic();
    int nb_files_expected = std::distance( fs::begin(fs::directory_iterator(data_path)), fs::end(fs::directory_iterator(data_path)) );
    tac();
    std::cout << " Expected files in directory : " << nb_files_expected << " in : " << duration() << " s" << std::endl;

// Open images
    int image_count = 0;
    tic();
    for( auto& p: fs::directory_iterator(data_path) )
        {
        image_count++;
        ProgressBar('P', 1.0*image_count / nb_files_expected);

        // get filename and store filename in the dataset
        std::string filename_str = p.path().string();
        Dataset.push_back(filename_str);
        }

// Randomize order
//     std::srand ( unsigned ( std::time(0) ) );  // ne pas utiliser pour garantir le meme ordre des données!
    std::random_shuffle ( Dataset.begin(), Dataset.end() );

    return image_count;
}

void ECHO::ReadInput(int offset, int num_images, float** inputs)
{
    if( Dataset.empty() ) { std::cout << " no dataset, use ReadPath first or check its output ! " << std::endl; return ; }

    int w,h,n;
    unsigned char *data;

// Copy desired number of input to inputs (and convert images)
    for( int j=offset; j<offset+num_images ; j++)
        {
        // read image
        data = stbi_load(Dataset[j].c_str(), &w, &h, &n, 0);

        // adapt the range to [0;1] and the type (float)
        for(int i=0; i<w*h; i++)
            inputs[j-offset][i] = 1.f * data[i] / 255.f;  // don't forget to take into account the offset
            //inputs[j-offset][i] = 1.f * data[i] / 128.f - 1.f;  // don't forget to take into account the offset
        // free space
        stbi_image_free(data);
        }
    tac();
    std::cout << " Total images read : " << num_images << " within : " << duration() << " s" << std:: endl;
}

void ECHO::ReadLabel(int offset,int num_labels, char** outputs)
{
    if( Dataset.empty() ) { std::cout << " no dataset, use ReadPath first or check its output ! " << std::endl; return ; }

    tic();
    for(int i=offset; i<offset+num_labels ; i++)
        {
        // get the string location of "CH_patient"
        std::string label_str = Dataset[i];
        std::size_t found = label_str.rfind("CH_patient");

        outputs[i-offset][0] = -1;  // error value

        // if pattern found
        if (found!=std::string::npos)
            {
            //  get the first 3 characters of the filename
            label_str = label_str.substr( found-1, 3);

            char label = ECHO::convert_label( label_str );
            // update the output for the two classes
            outputs[i-offset][0] = label;
            }
        if( outputs[i-offset][0] < 0)
            std::cout << " ERROR" << std::endl;
        }
    tac();
    std::cout << " Total images read : " << num_labels << " within : " << duration() << std:: endl;
}


char ECHO::convert_label(std::string const &s)
{
    if( s == "2CH") return 1;
    if( s == "4CH") return 0;
    return 0;
}

std::string ECHO::convert_label( char s)
{
    if( s == 1 ) return "2CH";
    if( s == 0 ) return "4CH";
    return "error";
}

//à quoi correspondent im, R et C ?

void ECHO::PrintImage(float *im, int R, int C)
{
    std::string m=" .,:;ox%#@";
    unsigned int l = m.length();

    for (int r = 0; r < R; r++)
        {
        std::string s="";
        for (int c = 0; c < C; c++)
            {
            int i = floor(im[(C*r) + c]*l); // à quoi correspond le contenu de i ?
            s+=m[i];
            }
        std::cout << s << std::endl;
        }
}
