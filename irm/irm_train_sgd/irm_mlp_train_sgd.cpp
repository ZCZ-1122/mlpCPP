#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <utility>
#include <thread>  // IF4
#include <mutex>   // IF4
#include <csignal> // IF4

#include "MLP_Network_SGD.h"
#include "MLP_Layer_SGD.h"
#include "IRM2D.h"
#include "timing_functions.h"
#include "progress_bar.h"
#include "gnuplot_utilities.h"

using namespace std;


// LINUX / MAC
//    const std::string SRC_PATH = "/home/tgrenier/Documents/Clanu21/cpp";

    const std::string SRC_PATH = "/home/zzhang/Bureau/cpp_examen";
    const std::string GNUPLOT_PATH="/usr/bin/gnuplot";

// WINDOWS
//    const std::string SRC_PATH = "C:/Users/Zhenchen Zhang/Desktop/cppV2";
//    const std::string GNUPLOT_PATH="C:/Program Files/gnuplot/bin/wgnuplot.exe";

//"C:\\\"Program Files\"\\gnuplot\\bin\\wgnuplot.exe"


// global variable for exchange information with handler
std::string *optim_log_filename_ptr;
int * maxEpoch_ptr;

// Protocoles
void ACCURACYandLOSS(MLP_Network_SGD *mlp,float **dataSet, int nDataSet,char **desiredOutputs, float *accuracy,float *loss);
void signal_callback_handler(int signum);


int main(int argc, char *argv[])
{
    cout << "SGD Train " << endl;

        cout << "argc = " << argc<< endl; //ZZC
        for(int i =0;i<argc;i++)          //ZZC
            cout << "argv" << i << " : " << argv[i]<< endl; //ZZC

    if( argc < 2)
        {
        cerr << " Usage : " << argv[0] << " file.bin" << endl;
        cerr << " where : file.bin is the file where the network's architecture and weights will be stored" << endl;
        return -1;
        }


    signal (SIGINT, signal_callback_handler);

    int nInputUnit      = 64*64;
    int nOutputUnit     = 9;
    vector<int> nHiddenUnit({16, 16}); // 40;
    int nHiddenLayer    = nHiddenUnit.size();

    float learningRate  = 0.0001;
    int maxEpoch        = 50;
    int nMiniBatch      = 50;

    maxEpoch_ptr = & maxEpoch;   // global variable update

    //IRM2D Input Array Allocation and Initialization
    IRM2D irm;


    cout << " -> Reading train directory : " << SRC_PATH+"/IRM2D/train" << endl;
    int nTrainingSet = irm.ReadPath(SRC_PATH+"/data/IRM2D/train");
    nTrainingSet = 5000;  //! Limit to 1000 images

    float errMinimum    = 0.001;

    //Allocate
    float **inputTraining		    = new float*[nTrainingSet];
    char **desiredOutputTraining	= new char*[nTrainingSet];

    for( int i = 0; i < nTrainingSet; i++ )
    {
        inputTraining[i]		    = new float[nInputUnit]{};
        desiredOutputTraining[i]	= new char[nOutputUnit]{};
    }
    irm.ReadInput(0, nTrainingSet, inputTraining);
    irm.ReadLabel(0, nTrainingSet, desiredOutputTraining);


    irm.SaveImage("a.png", inputTraining[0], 64,64 );

///////////////////////////////////////////////////////////////////////////////////
    cout << " \n \n" ;
    cout << " -> Reading test directory : " << SRC_PATH+"/IRM2D/test" << endl;
    int nTestSet = irm.ReadPath(SRC_PATH+"/data/IRM2D/test");
    nTestSet = 1000;

    float **inputTest			    = new float*[nTestSet];
    char **desiredOutputTest        = new char*[nTestSet];

    for(int i = 0;i < nTestSet;i++)
    {
        inputTest[i]			    = new float[nInputUnit];
        desiredOutputTest[i]    	= new char[nOutputUnit]{};
    }

    int sums=0;
    float accuracyRate=0.F;

    irm.ReadInput(0, nTestSet, inputTest);
    irm.ReadLabel(0, nTestSet, desiredOutputTest);


    MLP_Network_SGD mlp('L');
    mlp.Allocate(nInputUnit,nHiddenUnit,nOutputUnit,nTrainingSet);

    string fullname = SRC_PATH+"/models/"+argv[1];                          // with the line command, fullname = C:/Users/Zhenchen Zhang/Desktop/cpp/models/sgd/irm_sgd.bin
    cout << "Will write models and weights to : \n \t" <<fullname << endl;
    size_t lastindex = fullname.find_last_of(".");                          // lastindex = the index of the last appearance of "." in fullname
    string rawname = fullname.substr(0, lastindex);                         // rawname = C:/Users/Zhenchen Zhang/Desktop/cpp/models/sgd/irm_sgd
    cout << "  Rawname for intermediate saving : \n \t" << rawname << endl;
    string best_filename;

// LOG and GRAPH
    string adam_log_filename = rawname + ".dat";
    optim_log_filename_ptr = &adam_log_filename;  // update global variable (see line 34)
    cout << "  Filename for log : \n \t" << adam_log_filename << endl;
    fstream adam_log;
    adam_log.open(adam_log_filename.c_str(),fstream::out | fstream::trunc);// ouverture en écriture
    adam_log << "epoch \t train_{loss} \t test_{loss} \t lr \t train_{acc} \t  test_{acc} \t duration(s)" << endl;

// THREAD FOR REAL TIME UPDATE
    string gnuplot_graph_acc = SRC_PATH+"/models/graph_acc_adam.plt";
    GenerateGraphAcc(adam_log_filename, gnuplot_graph_acc, maxEpoch, true);
    string system_call_string(GNUPLOT_PATH+" "+gnuplot_graph_acc);
    std::thread gnuplot_thread_acc( system, system_call_string.c_str() );
    string gnuplot_graph_loss = SRC_PATH+"/models/graph_loss_adam.plt";
    GenerateGraphLoss(adam_log_filename, gnuplot_graph_loss, maxEpoch, true);
    system_call_string = GNUPLOT_PATH+" "+gnuplot_graph_loss;
    std::thread gnuplot_thread_loss( system, system_call_string.c_str() );

// Permutation vector
    std::vector<unsigned int> indexes(nTrainingSet);
    for(unsigned int i=0; i<indexes.size(); i++) indexes[i]=i;


    //Start clock
    clock_t start, finish;   // in fact, these are of type Long, since "typedef long clock_t"
    double elapsed_time;     // jingguo de shijian
    start = clock();
    float local_time = 0;

    float initialLR = learningRate;

    float maxTestAccuracyRate = 0;
    int epoch = 0;
    while (epoch < maxEpoch)
        {
        tic(); // start the chrono.
        std::srand ( unsigned ( std::time(0) ) );
        std::random_shuffle(indexes.begin(), indexes.end()); // indexes were like [0,1,2..,nTrainingSet-1], 这一行代码用来打乱indexes中元素
        ProgressBar('R'); // Reset jinDuTiao

        float sumError=0;
        int batchCount=0;
        for (int i = 0; i < nTrainingSet; i++)
            {

            if( (i%100) == 0 ) local_time = duration_from_tic(); // return time(in second) elapsed from last tic
            if( (i%10) == 0  )
                {
                ProgressBar('P', 1.0*i / nTrainingSet, string(" - " + std::to_string(local_time)).c_str(), 60); // Print a jinDuTiao in the terminal
                }

            mlp.ForwardPropagateNetwork(  inputTraining[indexes[i]]         );
            mlp.BackwardPropagateNetwork( desiredOutputTraining[indexes[i]] ); //
            sumError += mlp.LossFunction( desiredOutputTraining[indexes[i]] );

            if( ((batchCount+1) % nMiniBatch) == 0)
                {
                mlp.UpdateWeight(learningRate);
                batchCount=0;
                }
            batchCount++;
            }
        ProgressBar('C');

        sumError /= nTrainingSet;

//TEST ACCURACY and LOSS
        int sums=0;
        float LossTest=0.F;
        float AccuracyTest=0.F;
//        for( int i=0; i<nTestSet; i++)
//            {
//            mlp.ForwardPropagateNetwork(inputTest[i]);
//            sums += mlp.CalculateResult(desiredOutputTest[i]);
//            LossTest += mlp.LossFunction( desiredOutputTest[i] );
//            }
//        AccuracyTest = (sums / (float)nTestSet) * 100;
//        LossTest /= nTestSet;

//         ACCURACYandLOSS(MLP_Network_SGD *mlp,float **dataSet, int nDataSet,char **desiredOutputs, float *accuracy,float *loss)
        ACCURACYandLOSS(&mlp,inputTest,nTestSet,desiredOutputTest,&AccuracyTest,&LossTest); // fisrt call

//TRAIN ACCURACY and LOSS
        sums=0;
        float LossTrain=0.F;
        float AccuracyTrain=0.F;
//        for( int i=0; i<nTrainingSet; i++)
//            {
//            mlp.ForwardPropagateNetwork(inputTraining[i]);
//            sums += mlp.CalculateResult(desiredOutputTraining[i]);
//            LossTrain += mlp.LossFunction( desiredOutputTraining[i] );
//            }
//        AccuracyTrain = (sums / (float)nTrainingSet) * 100;
//    LossTrain /= nTrainingSet;
        ACCURACYandLOSS(&mlp,inputTraining,nTrainingSet,desiredOutputTraining,&AccuracyTrain,&LossTrain); // second call

// STOP CHRONO
        tac(); // tic() is at line 161

// DISPLAY AND SAVE LOG
        std::cout << std::fixed << std::setprecision(3);
        if(epoch%30 == 0) // we print the tilte below every 30 epochs
            cout << " epoch | Loss_tn | Loss_tt |     lr    |  ACC_tn  |  ACC_tt  |  duration(s)" << endl;

        cout << std::setw(6) << epoch << " | " << std::setw(7) << LossTrain << " | "<< std::setw(7) << LossTest
             << std::scientific << " | " << std::setw(8) << learningRate
             << std::fixed << " | " << std::setw(8) << AccuracyTrain << " | " << std::setw(8) << AccuracyTest
             << " | " << std::setw(8) << duration();
        adam_log << epoch<< "\t" <<LossTrain<< "\t" << LossTest << "\t" << learningRate
                 << "\t" << AccuracyTrain << "\t" << AccuracyTest << "\t" << duration() << endl;

// SAVE IF CURRENT IS THE BEST
        if(AccuracyTest > maxTestAccuracyRate)
            {
            maxTestAccuracyRate = AccuracyTest;
            best_filename = rawname + "_" + to_string(epoch) + "_" + to_string(AccuracyTest) + ".bin";
            std::ofstream os(best_filename, std::ofstream::binary); // output stream (save the best model)
            os<< mlp;
            os.close();
            cout << " * " << endl;
            }
        else cout << endl;

// EARLY STOP IF NO SIGNIFICANT CHANGE
        if (sumError < errMinimum)
            break;

// UPDATES FOR NEXT EPOCH
        learningRate = initialLR/(1+epoch*learningRate);    // learning rate progressive decay
        ++epoch;
    } //end of the While

    //Finish clock
    finish = clock();
    elapsed_time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"time: "<<elapsed_time<<" sec"<<endl; // this is the total time for all the epoches
    cout << " Best model : " << best_filename << endl;

    adam_log.close(); // closing a file is a must

    //saving best model
    MLP_Network_SGD mlp_best;
    std::ifstream is (best_filename, std::ifstream::binary); // firstly, we load the latest best model version whose name is "best_filename"
    is >> mlp_best;
    is.close();

    std::ofstream os_f(SRC_PATH+"/models/"+argv[1], std::ofstream::binary); // save the final best model to the file indicated by the command line
    os_f << mlp_best;
    os_f.close();


    // Training Set Result
    cout<<"[Result]"<<endl<<endl;
    sums=0;
    accuracyRate=0.F;
    float loss_not_used = 0.F;

//    for( int i=0; i<nTrainingSet; i++)
//        {
//        mlp_best.ForwardPropagateNetwork(inputTraining[i]);
//        sums += mlp_best.CalculateResult(desiredOutputTraining[i]);
//        }
//    accuracyRate = (sums / (float)nTrainingSet) * 100;
    ACCURACYandLOSS(&mlp_best,inputTraining,nTrainingSet,desiredOutputTraining,&accuracyRate,&loss_not_used); // third call

    cout << "[Training Set]\t"<<"Accuracy Rate: " << accuracyRate << " %"<<endl;

    // Test Set Result
    sums=0;
    accuracyRate=0.F;

//    for( int i=0; i<nTestSet; i++)
//        {
//        mlp_best.ForwardPropagateNetwork(inputTest[i]);
//        sums += mlp_best.CalculateResult(desiredOutputTest[i]);
//        }
//    accuracyRate = (sums / (float)nTestSet) * 100;
     ACCURACYandLOSS(&mlp_best,inputTest,nTestSet,desiredOutputTest,&accuracyRate,&loss_not_used); // fourth call

    cout << "[Test Set]\t"<<"Accuracy Rate: " << accuracyRate << " %"<<endl;


 // FREE ALLOCATED SPACES
    for( int i=0; i<nTrainingSet; i++)
        {
        delete [] desiredOutputTraining[i];
        delete [] inputTraining[i];
        }

    for( int i=0; i<nTestSet; i++)
        {
        delete [] desiredOutputTest[i];
        delete [] inputTest[i];
        }

    delete[] inputTraining;
    delete[] desiredOutputTraining;
    delete[] inputTest;
    delete[] desiredOutputTest;

    signal_callback_handler(0);

    return 0;
}

// fonction qui retournera les valeurs d'accuracy et de loss pour un jeu de donnees et un reseau
// Rq : mlp doit etre passe par adresse; "accuracy" "loss" also because these are two outputs
void ACCURACYandLOSS(MLP_Network_SGD *mlp,float **dataSet, int nDataSet,char **desiredOutputs, float *accuracy,float *loss)
{
    int sums = 0;
    for(int i=0;i<nDataSet;i++)
    {
        mlp->ForwardPropagateNetwork(dataSet[i]);
        sums += mlp->CalculateResult(desiredOutputs[i]);
        *loss += mlp->LossFunction(desiredOutputs[i]);
    }
    *accuracy = (sums/float(nDataSet)) * 100;
    *loss /= nDataSet;
}


// cette fonction est un garde fou pour limiter les conséquences de ceux qui ne lisent pas l'énoncé...
void signal_callback_handler(int signum)
{
   cout << "Caught signal " << signum << endl;

   system("killall gnuplot"); // will work only on linux

   string gnuplot_graph_acc = SRC_PATH+"/models/graph_acc_adam.plt";
   GenerateGraphAcc(*optim_log_filename_ptr, gnuplot_graph_acc, *maxEpoch_ptr, false);
   string system_call_string(GNUPLOT_PATH+" "+gnuplot_graph_acc);
   std::thread gnuplot_thread_acc( system, system_call_string.c_str() );

   string gnuplot_graph_loss = SRC_PATH+"/models/graph_loss_adam.plt";
   GenerateGraphLoss(*optim_log_filename_ptr, gnuplot_graph_loss, *maxEpoch_ptr, false);
   system_call_string = GNUPLOT_PATH+" "+gnuplot_graph_loss;
   std::thread gnuplot_thread_loss( system, system_call_string.c_str() );

   exit(signum);
}

