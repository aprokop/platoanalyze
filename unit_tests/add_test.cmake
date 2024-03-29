###############################################################################
## Plato_add_test( 
##    TEST_BASENAME == Base name of resulting unit test binary (arbitrary, but unique)
##    CPP_FILE      == cpp file containing unit tests
##    OPT_ARG_0     == if SKIP_MATH, unit test will not be compiled with math checking
## )
###############################################################################
function( Plato_add_test TEST_BASENAME CPP_FILE)

  if( ARGC GREATER 2 )  ## if optional argument included
    set(EXTRA_ARG ${ARGV2})
  else()  ## otherwise
    set(EXTRA_ARG)
  endif()

  message(STATUS "Adding unit tests: " ${TEST_BASENAME})

  set(Test_SOURCES ${CPP_FILE} PlatoUnitTestMain.cpp)

  set(Test_HEADERS PlatoTestHelpers.hpp)

  set(Test_NAME "${TEST_BASENAME}UnitTests")
  add_executable(${Test_NAME} ${Test_SOURCES} ${Test_HEADERS})

  target_link_libraries(${Test_NAME} analyzelib ${PLATO_LIBS} ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
  target_include_directories(${Test_NAME} PRIVATE "${PLATOENGINE_PREFIX}/include")

  if(NOT EXTRA_ARG STREQUAL "SKIP_MATH")
    target_compile_definitions(${Test_NAME} PRIVATE WATCH_ARITHMETIC)
  endif()

  target_include_directories(${Test_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/${Test_NAME})
  add_test(NAME run${Test_NAME} COMMAND ${ES_MPI_TEST})

endfunction(Plato_add_test)
###############################################################################
