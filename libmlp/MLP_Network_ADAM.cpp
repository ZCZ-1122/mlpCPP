///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#include "MLP_Network_ADAM.h"

#ifdef _OPENMP
    #include <omp.h>
#endif


void MLP_Network_ADAM::Allocate(int nInputUnit, vector<int> nHiddenUnit, int nOutputUnit, int nTrainingSet)
{
    MLP_Network<MLP_Layer_ADAM>::Allocate( nInputUnit,  nHiddenUnit,  nOutputUnit,  nTrainingSet);
    for (int i = 0; i < nHiddenLayer+1; i++)
        {
        layerNetwork[i].Beta1 = Beta1;
        layerNetwork[i].Beta2 = Beta2;
        }
}
