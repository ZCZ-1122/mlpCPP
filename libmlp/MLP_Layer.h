///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef MLP_Layer_H
#define MLP_Layer_H

#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

using FLOAT_TYPE = float;
using INTERNAL_FLOAT_TYPE = double;

//! Classe d'une couche du réseau
class  MLP_Layer
{
protected:
	//! nombre de neurones de la couche précédente
    int nPreviousNeurons;
	//! nombre de neurones de la couche courante
    int nCurrentNeurons;
    
	//! valeurs d'entrée de la couche courante
    FLOAT_TYPE* inputLayer;
    //! valeurs d'entrée de la couche courante
    FLOAT_TYPE* preActivationLayer;
    //!valeurs de sortie de la couche courante
    FLOAT_TYPE* outputLayer;
    //! poids des neurones
    FLOAT_TYPE* W;
    //! valeur du gradient des poids
    INTERNAL_FLOAT_TYPE* dW;
    //! pondération du gradient
    INTERNAL_FLOAT_TYPE* Delta;
    
	//! valeur du biais, un par neurone
    FLOAT_TYPE* b;
	//! valeur du gradient biais, un par neurone
    INTERNAL_FLOAT_TYPE* db;
	
	//! choix de la fonction d'activation
    char Activation_function;

public:
	//! contructeur
    MLP_Layer(char activation_function='R'):
        Activation_function(activation_function)
       { }
    //! destructeur
    virtual ~MLP_Layer()    {   Delete();   }
    
	//! Allocation de la couche courante
    virtual void Allocate(int previous_node_num, int nCurrentNeurons);
    //! désallocation de la couche courante
    virtual void Delete();
    
	//! passe avant
    virtual FLOAT_TYPE *ForwardPropagate(FLOAT_TYPE *inputLayer);
    //! passe arrière pour les couches cachées
    virtual void BackwardPropagateHiddenLayer(MLP_Layer* previousLayer);
    //! passe arrière pour la couche de sortie
    virtual void BackwardPropagateOutputLayer(char* desiredValues);
    
	//! fonction de mise à jour des poids
    virtual void UpdateWeight(FLOAT_TYPE learningRate)=0;
    
	//! fonction d'accès aux sorties de la couche courante
    FLOAT_TYPE* GetOutput() const  {   return outputLayer; }

    //! fonction de calcul de la fonction softmax pour les sorties de la couche courante (et accès aux valeurs par retour)
    FLOAT_TYPE* ComputeAndGetSoftmaxOutput();

    //! fonction d'accès aux poids
    FLOAT_TYPE* GetWeight() const  {   return W;           }
    //! fonction d'accès aux valeurs de delta
    INTERNAL_FLOAT_TYPE* GetDelta()  const  {   return Delta;       }
    //! fonction d'accès aux valeurs du gradient des poids
    INTERNAL_FLOAT_TYPE* GetWeightGradient()  const  {   return dW; }
    //! fonction d'accès aux valeurs du gradient des biais
    INTERNAL_FLOAT_TYPE* GetBiasGradient()  const  {   return db;   }

    //! fonction qui retroune le nombre de neurones de la couche courante
    int GetNumCurrent() const {   return nCurrentNeurons; }
    //! fonction qui retourne l'index du neurone ayant la valeur max en sortie
    int GetMaxOutputIndex() const;
    //! fonction pour couche mono neuronne (classification binaire) qui retourne 0 si la sortie du neurone < 0.5 et 1 sinon
    float GetBinaryOutput() const;
    //! fonction d'accès à la fonction d'activation choisie
    char GetActivationFunction() const    { return Activation_function; }
    //! fonction de modification de la fonction d'activation de la couche courante
    void SetActivationFunction(char a_f ) { Activation_function = a_f;  }

	//! calcul de la fonction d'activation
    inline FLOAT_TYPE ActivationFunction(INTERNAL_FLOAT_TYPE value, FLOAT_TYPE extra_value=0)
        {
        FLOAT_TYPE tmp(value);
        if (Activation_function == 'I')  { return tmp;     }                                // identity (no activation)
        if (Activation_function == 'S')  { return 1.0/(1.0 + exp(-tmp));     }             // Sigmoid
        if (Activation_function == 'R')  { if (tmp < 0) return 0.0; else return tmp;}      // ReLU
        if (Activation_function == 'L')  { if (tmp < 0) return 0.01*tmp; else return tmp;} // leaky ReLU ( [0.01:0.2] -> Xu 2015 )
//		if (Activation_function == 'E')  { if (tmp < 0) return 1.0507*1.67326*(exp(tmp) - 1); else return tmp;} // SELU = 1.0507*ELU_1.67326 ( Klambauer 2017)
        if (Activation_function == 's')  { return exp(tmp) / extra_value; }  // extra value will be the sum of exp (or one to compute the sum of exp)
        return 0;
        }

    //! calcul du gradient de la fonction d'activation à partir des outputLayer c-a-d la sortie
    inline FLOAT_TYPE DerivativeActivation(INTERNAL_FLOAT_TYPE value)
        {
        if (Activation_function == 'I')  { return 1.f;     }
        if (Activation_function == 'S') {return (1.f - value)*value; } // Sigmoid
/* The following is wrong for this project!
                {
                FLOAT_TYPE s_v = ActivationFunction(value);
                return (1.0 - s_v) * s_v;
                }
*/
        if (Activation_function == 'R')  { if( value <= 0) return -0.0; else return 1.0;}   // ReLU
        if (Activation_function == 'L')  { if( value <= 0) return -0.010; else return 1.0;} // leaky ReLU ( [0.01:0.2] -> Xu 2015 )
//        if (Activation_function == 'E')  { if (value <= 0) return 1.0507*1.67326*exp(value); else return 1.0;} // SELU = 1.0507*ELU_1.67326 ( Klambauer 2017)
        else return 0;
        }
    
    //! fonction de chargement d'une couche à partir d'un fichier
    friend ifstream& operator>>(ifstream& is, MLP_Layer& mlp_l);
	//! fonction de sauvegarde d'un couche dans un fichier
    friend ofstream& operator<<(ofstream& os, const MLP_Layer& mlp_l);
    
};

#endif
