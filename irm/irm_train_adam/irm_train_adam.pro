TEMPLATE = app

CONFIG += c++20 console
CONFIG += link_prl

TARGET = irm_mlp_train_adam

SOURCES = irm_mlp_train_adam.cpp

INCLUDEPATH += ../../libmlp
INCLUDEPATH += ../../libread

LIBS += -llibmlp -llibread -lstdc++fs
win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../../libmlp/release/  -L$$OUT_PWD/../../libread/release/
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../../libmlp/debug/ -L$$OUT_PWD/../../libread/debug/
else:unix: LIBS += -L$$OUT_PWD/../../libmlp/ -L$$OUT_PWD/../../libread/

message($$LIBS)
