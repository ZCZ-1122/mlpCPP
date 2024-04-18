///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef MLP_Network_H
#define MLP_Network_H


#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

/*! Classe de gestion du réseau Multi Layer Perceptron

*/
template<class LayerType>
class MLP_Network{
    
protected:
	//! Tableau des différentes couches (layer)
    LayerType *layerNetwork;
    
	//! Nombre de données pour l'entrainement
    int nTrainingSet;
	//! nombre de neurones d'entrée
    int nInputNeurons;
	//! nombre de neurones dans les couches cachées
    vector<int> nHiddenNeurons;
	//! nombre de neurones de sortie
    int nOutputNeurons;
	//! nombre de couches cachées
    int nHiddenLayer;

public:
    char Activation;

	//! constructeur, initialisation des champs
    MLP_Network(char activation='L'):
        nTrainingSet(-1),nInputNeurons(-1),nOutputNeurons(-1), Activation(activation)
        {layerNetwork = nullptr;}
    //! destructeur : libération de la mémoire
    virtual ~MLP_Network(){Delete();}

	//! fonction d'allocation du réseau
    virtual void Allocate(int nInputNeurons, vector<int> nHiddenNeurons, int nOutputNeurons, int nTrainingSet);
    //! fonction de désallocation du réseau
	virtual void Delete();
    
	//! fonction de calcul de la passe avant à partir de la couche \param inputNetwork
    virtual void ForwardPropagateNetwork(float* inputNetwork);
	//! fonction de calcul de la passe arrière, les labels sont passés à la fonction par \param desiredOutput
    virtual void BackwardPropagateNetwork(char *desiredOutput);
	//! fonction de mise à jour des poids du réseau, coefficienté par le taux d'apprentissage \param learningRate
    virtual void UpdateWeight(float learningRate);
	//! Calcul de la fonction de cout
    virtual float LossFunction(char *desiredOutput);
    //! Calcul de la précision
    virtual float CalculateResult(char *desiredOutput);

	//! fonction pour accéder aux couches du réseau
    virtual const LayerType * GetLayerNetwork() const {return this->layerNetwork;}

    //! fonction qui retourne la taille du jeu d'entrainement
    virtual int GetnTrainingSet() const { return this->nTrainingSet; }
    //! fonction qui retourne le nombre de neurones de la couche d'entrée (= premiere couche cachée)
    virtual int GetnInputUnit() const { return this->nInputNeurons; }
    //! fonction qui retourne le nombre de neurones dans les couches cachées
    virtual vector<int> GetnHiddenUnit() const { return this->nHiddenNeurons; }
    //! fonction qui retourne le nombre de neurones dans la couche de sortie
    virtual int GetnOutputUnit() const { return this->nOutputNeurons; }
    //! fonction qui retourne le nombre de couches cachées
    virtual int GetnHiddenLayer() const { return this->nHiddenNeurons.size(); }

	//! fonction de chargement d'un réseau à partir d'un fichier
    template<class T> friend ifstream& operator>>(ifstream& is, MLP_Network<T>& mlp);
	//! fonction de sauvegarde du réseau dans un fichier
    template<class T> friend ofstream& operator<<(ofstream& os, const MLP_Network<T>& mlp);
};




template<class LayerType>
void MLP_Network<LayerType>::Allocate(int nInputUnit, vector<int> nHiddenUnit, int nOutputUnit, int nTrainingSet)
{
    this->nTrainingSet     = nTrainingSet;
    this->nInputNeurons    = nInputUnit;
    this->nHiddenNeurons   = nHiddenUnit;
    this->nOutputNeurons   = nOutputUnit;
    this->nHiddenLayer     = nHiddenUnit.size();

    layerNetwork = new LayerType[nHiddenLayer+1];

    layerNetwork[0].Allocate(nInputUnit, nHiddenUnit[0]);
    for (int i = 1; i < nHiddenLayer; i++)
    {
        layerNetwork[i].Allocate(nHiddenUnit[i-1], nHiddenUnit[i]);
        layerNetwork[i].SetActivationFunction(Activation);
    }
    layerNetwork[nHiddenLayer].Allocate(nHiddenUnit[nHiddenLayer-1], nOutputUnit);
    if( nOutputUnit <= 2)
        layerNetwork[nHiddenLayer].SetActivationFunction('S');
    else
        layerNetwork[nHiddenLayer].SetActivationFunction('I');
}

template <class LayerType>
void MLP_Network<LayerType>::Delete()
{
    for (int i = 0; i < nHiddenLayer+1; i++)
    {
        layerNetwork[i].Delete();
    }
}

template <class LayerType>
void MLP_Network<LayerType>::ForwardPropagateNetwork(float* inputNetwork)
{
    float* outputOfHiddenLayer=NULL;

    outputOfHiddenLayer=layerNetwork[0].ForwardPropagate(inputNetwork);

   // ZZC
//    cout << " nHiddenLayer =  : "<< nHiddenLayer << endl;
//    cout << " nb neurunes de couche 1 = " << layerNetwork[0].GetNumCurrent() << endl;
//    cout << " fct d'activation couche 1 = " << layerNetwork[0].GetActivationFunction() << endl;
   // ZZC

    for (int i=1; i < nHiddenLayer ; i++)
    {
        outputOfHiddenLayer=layerNetwork[i].ForwardPropagate(outputOfHiddenLayer);                  //hidden forward
//        cout << " nb neurunes de couche " << i+1 << " = " << layerNetwork[i].GetNumCurrent() << endl;  // ZZC
//        cout << " fct d'activation couche " << i+1 << " = " << layerNetwork[i].GetActivationFunction() << endl;  // ZZC

    }
    layerNetwork[nHiddenLayer].ForwardPropagate(outputOfHiddenLayer);      // output forward
}

template <class LayerType>
void MLP_Network<LayerType>::BackwardPropagateNetwork(char* desiredOutput)
{
    layerNetwork[nHiddenLayer].BackwardPropagateOutputLayer(desiredOutput);  // back_propa_output
    for (int i= nHiddenLayer-1; i >= 0  ; i--)
        layerNetwork[i].BackwardPropagateHiddenLayer(&layerNetwork[i+1]);    // back_propa_hidden
}


template<class LayerType>
void MLP_Network<LayerType>::UpdateWeight(float learningRate)   // update weight according gradient and standart gradient descent
{
    for (int i = 0; i < nHiddenLayer; i++)
        {
        layerNetwork[i].UpdateWeight(learningRate);
        }
   layerNetwork[nHiddenLayer].UpdateWeight(learningRate);
}

template<class LayerType>
float MLP_Network<LayerType>::LossFunction(char* desiredOutput) //categorical cross entropy
{
    float *outputNetwork = layerNetwork[nHiddenLayer].ComputeAndGetSoftmaxOutput(); // GetOutput();
    float err=0.F;

    for (int j = 0; j < nOutputNeurons; ++j)
        //CCE
        if( (desiredOutput[j] == 1) && (outputNetwork[j] > 0) )
            err += desiredOutput[j] * log( outputNetwork[j] );

    err = -err;
    return err;
}

template <class LayerType>
float MLP_Network<LayerType>::CalculateResult(char* desiredOutput)
{
    int maxIdx = 0; // index of the max output(probaility)

    if( layerNetwork[nHiddenLayer].GetNumCurrent() > 2 ) // In our case, it is surely biger than 2 (2 corresponds to a 2-class classfication(cat or not)
        {
        maxIdx = layerNetwork[nHiddenLayer].GetMaxOutputIndex();
        if(desiredOutput[maxIdx] == 1)
            return 1; // this means that the prediction of the class is correct
        return 0;     // this means that the prediction of the class is wrong
        }
    else
        {
        if( layerNetwork[nHiddenLayer].GetBinaryOutput() == desiredOutput[0] )
            return 1;
        else
            return 0;
    }
}


template <class LayerType>
ifstream& operator>>(ifstream& is, MLP_Network<LayerType>& mlp)
{
    int nInputUnit, nOutputUnit, nHiddenLayer, tmp_nHiddenUnit;
    vector<int> nHiddenUnit;

    is.read( reinterpret_cast<char *>( &nInputUnit  ), sizeof(int) );
    is.read( reinterpret_cast<char *>( &nHiddenLayer), sizeof(int) );
    for(int i=0;i<nHiddenLayer;i++)
        {
        is.read( reinterpret_cast<char *>( &tmp_nHiddenUnit ), sizeof(int) );
        nHiddenUnit.push_back(tmp_nHiddenUnit);
        }
    is.read( reinterpret_cast<char *>( &nOutputUnit ), sizeof(int) );

    mlp.Allocate(nInputUnit, nHiddenUnit, nOutputUnit, 0);

    for (int i = 0; i < mlp.nHiddenLayer + 1; i++)
        is >> mlp.layerNetwork[i];

    return is;
}

template <class LayerType>
ofstream& operator<<(ofstream& os, const MLP_Network<LayerType>& mlp)
{
    os.write(reinterpret_cast<const char *>( &(mlp.nInputNeurons)   ), sizeof(int) );
    os.write(reinterpret_cast<const char *>( &(mlp.nHiddenLayer) ), sizeof(int) );
    for(int i=0;i<mlp.nHiddenLayer;i++)
        os.write( reinterpret_cast<const char *>( &(mlp.nHiddenNeurons[i]) ), sizeof(int) );
    os.write(reinterpret_cast<const char *>( &(mlp.nOutputNeurons)  ), sizeof(int) );

    for (int i = 0; i < mlp.nHiddenLayer + 1; i++)
        os << mlp.layerNetwork[i];

    return os;
}


#endif
