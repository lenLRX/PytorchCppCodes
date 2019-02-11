cmake_minimum_required (VERSION 3.1)
project(cppSimulator)

set(CMAKE_CXX_STANDARD 11)

#header only lib, just include headers
set(RAPIDJSON_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/SimDota2/thirdparty/rapidjson/include")

include_directories(include)
include_directories(${RAPIDJSON_INCLUDE_DIRS})
message("rapidjson: " ${RAPIDJSON_INCLUDE_DIRS})

file(GLOB HEADERS include/*.h)
file(GLOB HEADERS include/Ability/*.h)

aux_source_directory(src SRCS)
aux_source_directory(src/Ability SRCS)

add_definitions("-DSIMDOTA2_DLLEXPORT")
add_definitions("-D_GLIBCXX_USE_CXX11_ABI=0")

add_library(cppSimulator SHARED ${SRCS} ${HEADERS})

set_property(TARGET cppSimulator PROPERTY CXX_STANDARD 11)

if (WIN32)
    install(TARGETS cppSimulator RUNTIME DESTINATION bin)
else()
    install(TARGETS cppSimulator LIBRARY DESTINATION lib)
endif()
