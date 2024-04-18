// T. Grenier - Clanu 2021-2022 - INSA Lyon - GE

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#include "IRM2D.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>    // std::random_shuffle


#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "timing_functions.h"
#include "progress_bar.h"

/*
#ifdef __GNUC__
#  if __GNUC_PREREQ(9,0)
//      If  gcc_version >= 9.0
        #include <filesystem>
        namespace fs = std::filesystem;
#  else
//       Else
        #include <experimental/filesystem>
        namespace fs = std::experimental::filesystem;
#  endif
#else
    #include <filesystem>
    namespace fs = std::filesystem;
#endif
    */

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;


int IRM2D::ReadPath(std::string pathname)
{
    // Clear existing list of files
    Dataset.clear();

    std::cout << "Reading Directory : " << pathname << std::endl;
    fs::path data_path(pathname);

// Compute number of files in the directory
    tic();
    int nb_files_expected = std::distance( fs::begin(fs::recursive_directory_iterator(data_path)), fs::end(fs::recursive_directory_iterator(data_path)) );
    tac();
    std::cout << " Expected files in directory : " << nb_files_expected << " in : " << duration() << " s" << std::endl;

// Open images
    int image_count = 0;
    tic();
    for (auto const& p : fs::recursive_directory_iterator(data_path) )
        {
        ProgressBar('P', 1.0*image_count / nb_files_expected);

        // get filename and store filename in the dataset
        std::string filename_str = p.path().string();
        if( fs::is_regular_file( fs::status(filename_str) ) )  // verify that the file is a regular file (not a directory, link, pipe...)
            {
            Dataset.push_back(filename_str);
            image_count++;
            }
        }

    std::cout << std::endl;
    std::cout << " Files found in directory : " << image_count << std::endl;
// Randomize order
//     std::srand ( unsigned ( std::time(0) ) );  // ne pas utiliser pour garantir le meme ordre des données!
    std::random_shuffle ( Dataset.begin(), Dataset.end() );

    return image_count;
}

void IRM2D::ReadInput(int offset, int num_images, float** inputs)
{
    if( Dataset.empty() ) { std::cout << " no dataset, use ReadPath first or check its output ! " << std::endl; return ; }



// Copy desired number of input to inputs (and convert images)
    for( int j=offset; j<offset+num_images ; j++)
        {
        int w,h,n;
        unsigned char *data;

        ProgressBar('P', 1.0*j / num_images);
        // read image
        data = stbi_load(Dataset[j].c_str(), &w, &h, &n, 0);

        // adapt the range to [0;1] and to the type (float)
        for(int i=0; i<w*h; i++)
            inputs[j-offset][i] = 1.f * data[i] / 255.f;  // don't forget to take into account the offset

        // free space
        stbi_image_free(data);
        }
    tac();
    std::cout << std::endl;
    std::cout << " Total images read : " << num_images << " within : " << duration() << " s" << std:: endl;
}

void IRM2D::ReadLabel(int offset,int num_labels, char** outputs)
{
    if( Dataset.empty() ) { std::cout << " no dataset, use ReadPath first or check its output ! " << std::endl; return ; }

    tic();
    for(int i=offset; i<offset+num_labels ; i++)
        {
        ProgressBar('P', 1.0*i / num_labels);
        // get the string location of "CH_patient"
        std::string label_str = Dataset[i];
        std::string label_short = label_str.substr(label_str.size()-8, 4);

        if( convert_label(label_short) < 0)
            std::cout << " ERROR" << std::endl;
        outputs[i-offset][convert_label(label_short)] = 1;

        }
    tac();
    std::cout << std::endl;
    std::cout << " Total labels read : " << num_labels << " within : " << duration() << std:: endl;
}


char IRM2D::convert_label(std::string const &s)
{
    if( s == "PD-A") return 0;
    if( s == "PD-C") return 1;
    if( s == "PD-S") return 2;

    if( s == "T1-A") return 3;
    if( s == "T1-C") return 4;
    if( s == "T1-S") return 5;

    if( s == "T2-A") return 6;
    if( s == "T2-C") return 7;
    if( s == "T2-S") return 8;

    return -1;
}

std::string IRM2D::convert_label(char s)
{
    if( s == 0 ) return "PD-A";
    if( s == 1 ) return "PD-C" ;
    if( s == 2 ) return "PD-S";

    if( s == 3 ) return "T1-A";
    if( s == 4 ) return "T1-C";
    if( s == 5 ) return "T1-S";

    if( s == 6 ) return "T2-A";
    if( s == 7 ) return "T2-C";
    if( s == 8 ) return "T2-S";

    return "error";
}


//à quoi correspondent im, R et C ?
void IRM2D::PrintImage(float *im, int R, int C)
{
    std::string m=" .,:;ox%#@";
    //std::wstring m( " \u2591\u2592\u2593\u2588");
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

void IRM2D::SaveImage(const char *filename, float *im, int R, int C) const
{
    std::cout << "image saved"<< std::endl;

    unsigned char *im_uc = new unsigned char[R*C];
    for(int i=0; i<R*C; i++)
        im_uc[i] = (unsigned char) floor(im[i]*255);
    stbi_write_png(filename, C, R, 1, im_uc, C);
    delete[] im_uc;
    std::cout << "image saved"<< std::endl;
}

/*
void IRM2D::SaveDataSet(int offset, int num_images) const
{
    if( Dataset.empty() ) { std::cout << " no dataset, use ReadPath first or check its output ! " << std::endl; return ; }

    if( offset > Dataset.size() - 1 )
        offset = Dataset.size() - offset;

    int nb_to_display=num_images;
    if(offset + num_images > Dataset.size()-1 )
        nb_to_display = Dataset.size()-1 - offset;


    for(int i=offset; i<offset+nb_to_display ; i++)
}
*/
