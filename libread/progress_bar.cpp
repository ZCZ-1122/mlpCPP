// T. Grenier - Clanu 2017-2018-2019-2020-2021 - INSA Lyon - GE

///////////////////////////////////////////////////////////
//           DO NOT MODIFY THIS FILE                     //
///////////////////////////////////////////////////////////

#include "progress_bar.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <utility>
#include <algorithm>

void ProgressBar(char state, float current_progress, const char *extra_text, int barWidth)
{
    static int old_pos = 0;
    float progress;

    if(state == 'R') { old_pos = 0;  return; }  // RESET

    if(state == 'C')                            // CLEAR
//        { std::cout << "\r\e[K" << std::flush;  return; }  //  // Clear a whole line
           { std::cout << std::flush;  return; }  //  // Clear a whole line

    if(state == 'P')                            // PRINT
        {
        progress = current_progress;
        int pos = barWidth * progress;
        if( pos == old_pos) return;     // increase only each bar
        std::cout << "[";
        for (int i = 0; i < barWidth; ++i)
            {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
            }
        std::cout << "]" << extra_text << "\r";
        std::cout.flush();
        old_pos = pos;
//        if( pos == barWidth)  std::cout << std::endl;
        }
}
