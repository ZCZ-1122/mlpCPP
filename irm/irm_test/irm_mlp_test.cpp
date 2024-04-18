#include <iostream>
#include <string>

#include "MLP_Network_SGD.h"
#include "MLP_Network_ADAM.h"

#include "IRM2D.h"

#include "timing_functions.h"   //tic() & tac() &  duration() functions


#define MAX_FILENAME_LENGTH  100

using namespace std;


// Question 2.2.1
// LINUX / MAC
    const string SRC_PATH = "/home/zzhang/Bureau/cpp_examen";
// WINDOWS
//    const string SRC_PATH = "C:/Users/Zhenchen Zhang/Desktop/cppV2";

// "D:/Documents/_NextCloud/documents/Insa-GE/2021/Clanu/cpp";


int main(int argc, char *argv[])
{
#if defined(_OPENMP)
    cout << " OPENMP is activated : great! " << endl;
#else
    cout << " OPENMP is not activated (good for debug)" << endl;
#endif

#ifdef __FAST_MATH__
    cout << " fast-math is activated : great! " << endl;
#else
    cout << " fast-math is strangely not activated " << endl;
#endif

    if( argc < 2)
        {
        cerr << " Usage : " << argv[0] << " file.bin" << endl;
        cerr << " where : file.bin is the architecture and weights of the network to be loaded" << endl;
        return -1;
        }

// Load Model
    cout << "ZZC hello " << endl;
    cout << "argc = " << argc<< endl; //ZZC
    for(int i =0;i<argc;i++)          //ZZC
        cout << "argv" << i << " : " << argv[i]<< endl; //ZZC


    cout << "Reading network models (architecture and weights) from : " << SRC_PATH << "/models/" << argv[1] << endl;
    tic();
    MLP_Network_ADAM mlp;
    std::ifstream is (SRC_PATH+"/models/"+argv[1], std::ifstream::binary);
    is >> mlp;
    is.close();
    tac();
    cout << " Model loaded in : "<< duration() << " s" << endl;

    // 2022.06.09
    cout << endl << endl;
    cout << "2020 06 09" << endl;
    cout << "nHiddenLayer : " << mlp.GetnHiddenLayer()<<endl;
    for(int i =0 ; i<mlp.GetnHiddenLayer();i++)
    {
        cout <<mlp.GetnHiddenUnit()[i] << endl;
    }


// Reading Images
    int nInputUnit      = 64*64;
    int nOutputUnit     = 9;


    //Input Array Allocation and Initialization
    IRM2D irm2d;

    cout << " Reading directory : " << SRC_PATH+"/data/IRM2D" << "  ";

    int nTestSet = irm2d.ReadPath(SRC_PATH+"/data/IRM2D/test");
    nTestSet = 1000;
    cout <<  nTestSet << endl;

    //Allocate
    float **inputTest			= new float*[nTestSet];
    char **desiredOutputTest	= new char*[nTestSet];

    for(int i = 0;i < nTestSet;i++)
        {
        inputTest[i]			= new float[nInputUnit]{};
        desiredOutputTest[i]	= new char[nOutputUnit]{};
        }


    // ReadData (images and labels)
    irm2d.ReadInput(0, nTestSet, inputTest);
    irm2d.ReadLabel(0, nTestSet, desiredOutputTest);


    tic();
    //TEST ACCURACY and LOSS
    int sums=0;
    float LossTest=0.F;
    float AccuracyTest=0.F;
    for( int i=0; i<nTestSet; i++)
        {
        mlp.ForwardPropagateNetwork(inputTest[i]);
        sums += mlp.CalculateResult(desiredOutputTest[i]);
        LossTest += mlp.LossFunction( desiredOutputTest[i] ); // Loss function = Fonction de cout
        }
    AccuracyTest = (sums / (float)nTestSet) * 100;
    LossTest /= nTestSet;
    tac();
    cout << "[Test Set]\t Loss : "<< LossTest << " Accuracy : " << AccuracyTest << " %"<< "   (compute time : "<< duration() << ")" << endl;


//ADD FROM HERE THE CODE THAT PRINT ALL ERRONEOUS PREDICTED IMAGE INDEXES (Question 2.3.2)

// Question 2.3 (2):
//    int nbErrorPredictedImage = 0;

//    cout << "Indices of all the erroneous predicted images : " <<endl;
//    for(int i = 0;i < nTestSet;i++)
//        {
//         mlp.ForwardPropagateNetwork(inputTest[i]);
//         int maxIndex = mlp.GetLayerNetwork()[mlp.GetnHiddenLayer()].GetMaxOutputIndex();

//         if(desiredOutputTest[i][maxIndex] == 0)
//         {
//             nbErrorPredictedImage ++;
//             cout << i << endl;
//         }
//        }
//    cout << "compte : " << nbErrorPredictedImage << endl;



// Question 2.3 (3):
//    cout << "############################### " << endl;

//    // Firstly, we inject a data sample(the first one) into the network
//    mlp.ForwardPropagateNetwork(inputTest[0]);

//    // Display output values of the network
//    float* outputOfNetwork = mlp.GetLayerNetwork()[mlp.GetnHiddenLayer()].GetOutput();

//    cout << "Display the output : " << endl;

//    for (int i=0;i<9;i++)
//    {
//        cout << outputOfNetwork[i] << endl;
//    }

//    // Deduce the class from the network output
//    int maxIndice = mlp.GetLayerNetwork()[mlp.GetnHiddenLayer()].GetMaxOutputIndex(); \
//    cout << " maxIndice : " << maxIndice << endl;
//    cout << " predicted : " << IRM2D::convert_label(maxIndice) << "  -  "<<endl;  //affichage prédiction

//    cout << "############################### " << endl;
//################################################################

//STOP HERE THE CODE THAT PRINT ALL ERRONEOUS PREDICTED IMAGE INDEXES (no more changes after this line)


//// Ask for an image and print it until -1
    int ind_im;
    cout << " which image ? ";
    cin >> ind_im;

while(ind_im != -1)
    {
    IRM2D::PrintImage(inputTest[ind_im], 64, 64);
    mlp.ForwardPropagateNetwork(inputTest[ind_im]);
    int Predicted = mlp.GetLayerNetwork()[mlp.GetnHiddenLayer()].GetMaxOutputIndex();    // récuperation de la valeur prédite par le réseau
    cout << " predicted : " << IRM2D::convert_label(Predicted) << "  -  "; //affichage prédiction et nom du fichier

    // print filename
    cout << irm2d.Dataset[ind_im] << endl;

    cout << " which image ? (-1 to exit) ";
    cin >> ind_im;
    }



    // Ask for an image and SAVE it until -1
//        int ind_im;
//        cout << " which image to save ? ";
//        cin >> ind_im;

//    while(ind_im != -1)
//        {
//        string filename = "C:/Users/Zhenchen Zhang/Desktop/cpp/image.png";
//        irm2d.SaveImage(filename.c_str(),inputTest[ind_im], 64, 64);

//        cout << irm2d.Dataset[ind_im] << endl;

//        cout << " which image to save ? (-1 to exit) ";
//        cin >> ind_im;
//        }

    for(int i=0; i<nTestSet; i++)
        {
        delete [] desiredOutputTest[i];
        delete [] inputTest[i];
        }

    delete[] inputTest;
    delete[] desiredOutputTest;
    
    return 0;
}
