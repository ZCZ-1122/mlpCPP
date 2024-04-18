TEMPLATE = lib
TARGET = libmlp

CONFIG += staticlib c++20 warn_off
CONFIG += create_prl
CONFIG -= qt

SOURCES = MLP_Layer.cpp MLP_Network.cpp \
	MLP_Layer_SGD.cpp MLP_Network_SGD.cpp \
	MLP_Layer_ADAM.cpp MLP_Network_ADAM.cpp \
	
HEADERS = MLP_Layer.h MLP_Network.h \
    MLP_Layer_SGD.h   MLP_Network_SGD.h  \
    MLP_Layer_ADAM.h   MLP_Network_ADAM.h\

LIBS += -lstdc++fs


