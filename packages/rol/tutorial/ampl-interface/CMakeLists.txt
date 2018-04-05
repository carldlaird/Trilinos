INCLUDE_DIRECTORIES(/Users/cdlaird/Documents/software/ampl/solvers/)

TRIBITS_ADD_EXECUTABLE(
  Ampl-Interface
  SOURCES AmplOptimizationProblem.cpp
  ADD_DIR_TO_NAME
  )
