///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#include "CT2D.h"

#include <iostream>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void CT2D::ReadInput(std::string pathname,int num_images, float** inputs)
{
 /*   ifstream image_file(filename, ios::binary); // read MNIST-image database
    cout << "Reading Training Directory : " << filename << endl;
    if (image_file.is_open())
    {
        cout << "MNIST : IMAGE OPENED" << endl;
        int magic_number = 0;
        int num_images_from_file = 0;
        int image_rows = 0;
        int image_cols = 0;
        image_file.read((char*)&magic_number, sizeof(magic_number)); // get magic number : don't use
        magic_number = BytetoInt(magic_number);
        image_file.read((char*)&num_images_from_file, sizeof(num_images_from_file)); // get num_images : don't use
        num_images_from_file = BytetoInt(num_images_from_file);
        image_file.read((char*)&image_rows, sizeof(image_rows)); // get rows
        image_rows = BytetoInt(image_rows);
        image_file.read((char*)&image_cols, sizeof(image_cols)); // get cols
        image_cols = BytetoInt(image_cols);
        for (int i = 0; i < num_images; i++)
        {
            for (int r = 0; r < image_rows; r++)
            {
                for (int c = 0; c < image_cols; c++)
                {
                    unsigned char temp = 0;
                    image_file.read((char*)&temp, sizeof(temp));
                    if ((float)temp > 0)
                        inputs[i][(image_rows*r) + c] = 1.f;
                    else
                        inputs[i][(image_rows*r) + c] = 0.f;
                }
            }
        }
    }
    else
       cout << "MNIST : IMAGE NOT OPENED" << endl;
*/
}

void CT2D::ReadLabel(std::string filename,int num_labels, float** outputs)
{
 /*   ifstream label_file(filename, ios::binary); // read MNIST-label database
    cout << "Reading Label file : " << filename << endl;
    if (label_file.is_open()){
        cout << "MNIST : LABEL OPENED" << endl;
        int magic_number = 0;
        int num_labels_from_file = 0;
        label_file.read((char*)&magic_number, sizeof(magic_number)); // get magic number : don't use
        magic_number = BytetoInt(magic_number);
        label_file.read((char*)&num_labels_from_file, sizeof(num_labels_from_file)); // get num_images : don't use
        num_labels_from_file = BytetoInt(num_labels_from_file);
        for (int i = 0; i < num_labels; i++){
            unsigned char temp = 0;
            label_file.read((char*)&temp, sizeof(temp));
            for (int j = 0; j < 10; j++){
                if ((int)temp == j)
                    outputs[i][j] = 1.f;
                else
                    outputs[i][j] = 0.f;
            }
        }
    }
    else
        cout << "MNIST : LABEL NOT OPENED" << endl;
*/
}


char CT2D::convert_label(std::string const &s)
{
    if( s == "T1-A") return 1;
    if( s == "T1-S") return 2;
    if( s == "T1-C") return 3;

    if( s == "T2-A") return 4;
    if( s == "T2-S") return 5;
    if( s == "T2-C") return 6;

    if( s == "CT-A") return 7;
    if( s == "CT-S") return 8;
    if( s == "CT-C") return 9;

    return 0;
}
