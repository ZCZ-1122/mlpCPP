///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef MLP_Network_ADAM_H
#define MLP_Network_ADAM_H

#include "MLP_Network.h"
#include "MLP_Layer_ADAM.h"


#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

/*! Classe de gestion du réseau Multi Layer Perceptron

*/
class MLP_Network_ADAM: public MLP_Network <MLP_Layer_ADAM> {

public:
    double Beta1;
    double Beta2;

	//! constructeur, initialisation des champs
    MLP_Network_ADAM(double beta1=0.9, double beta2=0.99, char activation='L'):MLP_Network<MLP_Layer_ADAM>(activation),
         Beta1(beta1), Beta2(beta2) {}
    //! destructeur : libération de la mémoire
    virtual ~MLP_Network_ADAM()     {}

	//! fonction d'allocation du réseau
    void Allocate(int nInputNeurons, vector<int> nHiddenNeurons, int nOutputNeurons, int nTrainingSet);

};
#endif
