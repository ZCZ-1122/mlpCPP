// T. Grenier - Clanu 2017-2018-2019-2020-2021 - INSA Lyon - GE

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef _progress_bar_functions_h
#define _progress_bar_functions_h

#include <string>


//! Display a progress bar 'P' if for normal Print, 'R' is for Reset, 'C' for clearing a line
void ProgressBar(char state, float current_progress=0, const char *extra_text=" ", int barWidth = 70 );

#endif 
