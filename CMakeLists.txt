cmake_minimum_required (VERSION 3.8)
message("prefix" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

add_subdirectory("SimDota2")

include_directories(.)
set(RAPIDJSON_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/SimDota2/thirdparty/rapidjson/include")
include_directories(${RAPIDJSON_INCLUDE_DIRS})
add_executable (A3CMain 
"A3CMain.cpp" "ActorCritic.h"
"A3CEnv.h" "A3CEnv.cpp"
"RewardNetwork.h"
"util.h")

target_link_libraries(A3CMain "${TORCH_LIBRARIES}" cppSimulator)
set_property(TARGET A3CMain PROPERTY CXX_STANDARD 11)

add_executable (Mini 
"minibatch.cpp" "ActorCritic.h"
"MiniEnv.h" "MiniEnv.cpp"
"RewardNetwork.h" "ReplayBuffer.h"
"util.h")

target_link_libraries(Mini "${TORCH_LIBRARIES}" cppSimulator)
set_property(TARGET Mini PROPERTY CXX_STANDARD 11)

add_executable (GAIL 
"GAIL/util.h"
"GAIL/Actor.h" "GAIL/DisCriminator.h"
"GAIL/Env.cpp" "GAIL/Env.h"
"GAIL/Critic.h" "GAIL/main.cpp")

target_link_libraries(GAIL "${TORCH_LIBRARIES}" cppSimulator)
set_property(TARGET GAIL PROPERTY CXX_STANDARD 11)

install(TARGETS A3CMain Mini GAIL RUNTIME DESTINATION bin)

add_executable (GAIL_MT 
"GAIL/util.h"
"GAIL/Actor.h" "GAIL/DisCriminator.h"
"GAIL/Env.cpp" "GAIL/Env.h"
"GAIL/Critic.h" "GAIL/main_mt.cpp")

target_link_libraries(GAIL_MT "${TORCH_LIBRARIES}" cppSimulator)
set_property(TARGET GAIL_MT PROPERTY CXX_STANDARD 11)

install(TARGETS A3CMain Mini GAIL_MT RUNTIME DESTINATION bin)
