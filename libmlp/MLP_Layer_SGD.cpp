///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#include "MLP_Layer_SGD.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

void MLP_Layer_SGD::UpdateWeight(FLOAT_TYPE learningRate)
{
//////////////// W
#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (int j = 0; j < nCurrentNeurons; j++)
        for (int i = 0; i < nPreviousNeurons; i++)
            W[j*nPreviousNeurons + i] +=  -learningRate * dW[j*nPreviousNeurons + i];

//////////////// b
#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (int j = 0; j < nCurrentNeurons; j++)
        b[j] += -learningRate * db[j];

// Clear gradient dW and db
#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (int j = 0; j < nCurrentNeurons; j++)
        for (int i = 0; i < nPreviousNeurons; i++)
            dW[j*nPreviousNeurons + i] = 0;

#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (int j = 0; j < nCurrentNeurons; j++)
        db[j]=0;
}

