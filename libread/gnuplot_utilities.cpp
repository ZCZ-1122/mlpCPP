// T. Grenier - Clanu 2017-2018-2019-2020-2021 - INSA Lyon - GE

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#include "gnuplot_utilities.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <utility>
#include <algorithm>



void GenerateGraphAcc(std::string &path_to_data, std::string &filename, int max_epcohs, bool replot)
{
    std::fstream graph;
    std::string tmp = path_to_data;
    std::replace( tmp.begin(), tmp.end(), '\\', '/'); // replace all '\\' by '/'
    graph.open(filename.c_str(), std::fstream::out | std::fstream::trunc);// ouverture en écriture
    if(replot == true )
        {
        graph   << "pause 5" << std::endl
                << "set key autotitle columnhead" << std::endl
                << "set xrange [0:"<<max_epcohs << "]" << std::endl
                << "set yrange [20:100]" << std::endl
                << "plot \""<< tmp << "\" using 1:5 with lines, "
                << "\""<< tmp << "\" using 1:6 with lines  " << std::endl
                << "do for [t=0:60]  { " << std::endl
                << "replot " << std::endl
                << "pause 2" << std::endl
                << "}"
                << std::endl;
        }
     else
        {
        graph   << "set key autotitle columnhead" << std::endl
                << "set xrange [0:"<<max_epcohs << "]" << std::endl
                << "set yrange [20:100]" << std::endl
                << "plot \""<< tmp << "\" using 1:5 with lines, "
                << "\""<< tmp << "\" using 1:6 with lines  " << std::endl
                << "pause -1" << std::endl
                << std::endl;
        }

     graph.close();
}


void GenerateGraphLoss(std::string &path_to_data, std::string &filename, int max_epcohs, bool replot)
{
    std::fstream graph;
    std::string tmp = path_to_data;
    std::replace( tmp.begin(), tmp.end(), '\\', '/'); // replace all '\\' by '/'
    graph.open(filename.c_str(), std::fstream::out | std::fstream::trunc);// ouverture en écriture
    if(replot == true )
        {
        graph   << "pause 5" << std::endl
                << "set key autotitle columnhead" << std::endl
                << "set xrange [0:"<<max_epcohs << "]" << std::endl
                << "set yrange [0:3]" << std::endl
                << "plot \""<< tmp << "\" using 1:2 with lines, "
                << "\""<< tmp << "\" using 1:3 with lines  " << std::endl
                << "do for [t=0:60] { " << std::endl
                << "replot " << std::endl
                << "pause 2" << std::endl
                << "}"
                << std::endl;
        }
    else
        {
        graph   << "set key autotitle columnhead" << std::endl
                << "set xrange [0:"<<max_epcohs << "]" << std::endl
                << "set yrange [0:3]" << std::endl
                << "plot \""<< tmp << "\" using 1:2 with lines, "
                << "\""<< tmp << "\" using 1:3 with lines  " << std::endl
                << "pause -1" << std::endl
                << std::endl;
        }


    graph.close();
}


