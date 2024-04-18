///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef MLP_Network_SGD_H
#define MLP_Network_SGD_H

#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>

#include "MLP_Network.h"
#include "MLP_Layer_SGD.h"

using namespace std;

/*! Classe de gestion du réseau Multi Layer Perceptron

*/
class MLP_Network_SGD: public MLP_Network <MLP_Layer_SGD>
{
public:
    //! constructeur, initialisation des champs
    MLP_Network_SGD( char activation='L'):MLP_Network<MLP_Layer_SGD>(activation)
		{}
    //! destructeur : libération de la mémoire
    virtual ~MLP_Network_SGD() 
		{}
};
#endif
