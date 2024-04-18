TEMPLATE = app

CONFIG += c++20 console
CONFIG += link_prl

TARGET = irm_mlp_test
SOURCES = irm_mlp_test.cpp
INCLUDEPATH += ../../libmlp
INCLUDEPATH += ../../libread

LIBS += -llibmlp -llibread -lstdc++fs
win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../../libmlp/release/  -L$$OUT_PWD/../../libread/release/
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../../libmlp/debug/ -L$$OUT_PWD/../../libread/debug/
else:unix: LIBS += -L$$OUT_PWD/../../libmlp/ -L$$OUT_PWD/../../libread/

message($$LIBS)

#win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../libmlp/release/liblibmlp.a
#else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../libmlp/debug/liblibmlpd.a
#else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../libmlp/release/libmlp.lib
#else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../libmlp/debug/libmlpd.lib
#else:unix: PRE_TARGETDEPS += $$OUT_PWD/../../libmlp/liblibmlp.a
