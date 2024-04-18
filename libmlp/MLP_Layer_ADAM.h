///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef MLP_Layer_ADAM_H
#define MLP_Layer_ADAM_H

#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

#include "MLP_Layer.h"

//! Classe d'une couche du réseau
class  MLP_Layer_ADAM : public MLP_Layer
{
protected:
    INTERNAL_FLOAT_TYPE* MW_next;
    INTERNAL_FLOAT_TYPE* Mb_next;
    INTERNAL_FLOAT_TYPE* SW_next;
    INTERNAL_FLOAT_TYPE* Sb_next;

    INTERNAL_FLOAT_TYPE* MW;
    INTERNAL_FLOAT_TYPE* SW;

    INTERNAL_FLOAT_TYPE* Mb;
    INTERNAL_FLOAT_TYPE* Sb;

public:
    INTERNAL_FLOAT_TYPE Beta1;
    INTERNAL_FLOAT_TYPE Beta2;
    int T;
    INTERNAL_FLOAT_TYPE Epsilon;

	//! contructeur
    MLP_Layer_ADAM(INTERNAL_FLOAT_TYPE beta1=0.9, INTERNAL_FLOAT_TYPE beta2=0.99, char activation_function='L'):
        MLP_Layer(activation_function),
        Beta1(beta1), Beta2(beta2), T(1), Epsilon(0.0000001)
       { MW = nullptr; Mb = nullptr; SW = nullptr; Sb = nullptr; }
    //! destructeur
    virtual ~MLP_Layer_ADAM()    {   Delete();   }
    
	//! Allocation de la couche courante
    void Allocate(int previous_node_num, int nCurrentNeurons);
    //! désallocation de la couche courante
    virtual void Delete();
     
	//! fonction de mise à jour des poids
    void UpdateWeight(FLOAT_TYPE learningRate);      
};

#endif
