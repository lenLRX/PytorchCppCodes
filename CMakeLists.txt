cmake_minimum_required (VERSION 3.8)
message("prefix" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

add_subdirectory("SimDota2")

include_directories(.)
set(RAPIDJSON_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/SimDota2/thirdparty/rapidjson/include")
set(SPDLOG_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/SimDota2/thirdparty/spdlog/include")
include_directories(${RAPIDJSON_INCLUDE_DIRS})
include_directories(${SPDLOG_INCLUDE_DIRS})

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

add_executable (GAIL_Tree 
"GAIL_Tree/util.h"
"GAIL_Tree/Actor.h" "GAIL_Tree/DisCriminator.h"
"GAIL_Tree/Env.cpp" "GAIL_Tree/Env.h"
"GAIL_Tree/Models.h" "GAIL_Tree/Models.cpp"
"GAIL_Tree/ActionChoice.h" "GAIL_Tree/ActionChoice.cpp"
"GAIL_Tree/LastHitMode.h" "GAIL_Tree/LastHitMode.cpp"
"GAIL_Tree/MovingMode.h" "GAIL_Tree/MovingMode.cpp"
"GAIL_Tree/Critic.h" "GAIL_Tree/main.cpp" GAIL_Tree/Trainer.h GAIL_Tree/Trainer.cpp)

target_link_libraries(GAIL_Tree "${TORCH_LIBRARIES}" cppSimulator)
set_property(TARGET GAIL_Tree PROPERTY CXX_STANDARD 11)

install(TARGETS A3CMain Mini GAIL GAIL_MT GAIL_Tree RUNTIME DESTINATION bin)
