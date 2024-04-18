// T. Grenier - Clanu 2017-2018-2019-2020-2021 - INSA Lyon - GE

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#ifndef _gnuplot_utilities_h
#define _gnuplot_utilities_h

#include <string>

void GenerateGraphAcc(std::string &path_to_data, std::string &filename, int max_epcohs, bool replot=true);
void GenerateGraphLoss(std::string &path_to_data, std::string &filename, int max_epcohs, bool replot=true);

#endif 
