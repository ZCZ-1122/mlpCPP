///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef MLP_Layer_SGD_H
#define MLP_Layer_SGD_H

#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>

#include "MLP_Layer.h"

using namespace std;

//! Classe d'une couche du réseau
class  MLP_Layer_SGD : public MLP_Layer
{  
public:
	//! contructeur
    MLP_Layer_SGD(char activation_function='R'):MLP_Layer(activation_function) { }

	//! fonction de mise à jour des poids
    virtual void UpdateWeight(FLOAT_TYPE learningRate);
};

#endif
