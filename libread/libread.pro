TEMPLATE = lib
TARGET = libread

CONFIG += staticlib c++20 warn_off
CONFIG += create_prl
CONFIG -= qt

SOURCES = IRM2D.cpp \
        progress_bar.cpp gnuplot_utilities.cpp timing_functions.cpp
      
        
HEADERS = IRM2D.h  \
        progress_bar.h  stb_image.h stb_image_write.h gnuplot_utilities.h timing_functions.h

LIBS += -lstdc++fs


