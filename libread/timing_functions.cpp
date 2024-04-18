// T. Grenier - Clanu 2017-2018-2019-2020-2021 - INSA Lyon - GE

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#include "timing_functions.h"

//! Start the chronometer.
void tic()
{
tic_time = std::chrono::steady_clock::now();
}

//! Stop the chronometer.
void tac()
{
tac_time = std::chrono::steady_clock::now();
}

//! Return time in seconds elapsed between last tic() and tac() calls.
double duration()
{
    std::chrono::duration<double> time_span = std::chrono::duration_cast< std::chrono::duration<double> >(tac_time - tic_time);
    return time_span.count();
}

//! Return time in seconds elapsed between last tic().
double duration_from_tic()
{
    std::chrono::duration<double> time_span = std::chrono::duration_cast< std::chrono::duration<double> >(std::chrono::steady_clock::now() - tic_time);
    return time_span.count();
}

//! Return time in seconds elapsed between last tac()
double duration_from_tac()
{
    std::chrono::duration<double> time_span = std::chrono::duration_cast< std::chrono::duration<double> >(std::chrono::steady_clock::now() - tac_time);
    return time_span.count();
}
