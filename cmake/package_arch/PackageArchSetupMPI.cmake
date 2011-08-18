# @HEADER
# ************************************************************************
#
#            Trilinos: An Object-Oriented Solver Framework
#                 Copyright (2001) Sandia Corporation
#
#
# Copyright (2001) Sandia Corporation. Under the terms of Contract
# DE-AC04-94AL85000, there is a non-exclusive license for use of this
# work by or on behalf of the U.S. Government.  Export of this program
# may require a license from the United States Government.
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the Corporation nor the names of the
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# NOTICE:  The United States Government is granted for itself and others
# acting on its behalf a paid-up, nonexclusive, irrevocable worldwide
# license in this data to reproduce, prepare derivative works, and
# perform publicly and display publicly.  Beginning five (5) years from
# July 25, 2001, the United States Government is granted for itself and
# others acting on its behalf a paid-up, nonexclusive, irrevocable
# worldwide license in this data to reproduce, prepare derivative works,
# distribute copies to the public, perform publicly and display
# publicly, and to permit others to do so.
#
# NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT
# OF ENERGY, NOR SANDIA CORPORATION, NOR ANY OF THEIR EMPLOYEES, MAKES
# ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
# RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY
# INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS
# THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
#
# ************************************************************************
# @HEADER

INCLUDE(AdvancedSet)
INCLUDE(MultilineSet)
INCLUDE(PrintNonemptyVar)
INCLUDE(FindProgramPlus)
INCLUDE(TPLDeclareLibraries)


FUNCTION(PACKAGE_ARCH_EXTRACT_BASE_DIR FILE_PATH BASE_DIR)
  IF (NOT ${BASE_DIR})
    GET_FILENAME_COMPONENT( ${BASE_DIR} ${FILE_PATH} PATH )
    IF (${PROJECT_NAME}_VERBOSE_CONFIGURE)
      PRINT_VAR(${BASE_DIR})
    ENDIF()
    SET(${BASE_DIR} ${${BASE_DIR}} PARENT_SCOPE)
  ENDIF()
ENDFUNCTION()


MACRO(PACKAGE_ARCH_FIND_MPI_COMPILER LANG)
  IF (${PROJECT_NAME}_ENABLE_${LANG})
    IF (CMAKE_${LANG}_COMPILER)
      MESSAGE(STATUS "Leaving current CMAKE_${LANG}_COMPILER="
        "${CMAKE_${LANG}_COMPILER} since it is already set!")
    ELSE()
      FIND_PROGRAM_PLUS(
        MPI_${LANG}_COMPILER
        NAMES ${ARGN}
        PATHS ${MPI_BIN_DIR_PATHS}
        )
      PACKAGE_ARCH_EXTRACT_BASE_DIR(${MPI_${LANG}_COMPILER} MPI_BASE_DIR)
      MESSAGE(STATUS "Setting CMAKE_${LANG}_COMPILER=\${MPI_${LANG}_COMPILER}")
      SET(CMAKE_${LANG}_COMPILER "${MPI_${LANG}_COMPILER}"
        CACHE FILEPATH
        "${LANG} compiler overridden by MPI_${LANG}_COMPILER")
      PRINT_VAR(CMAKE_${LANG}_COMPILER)
    ENDIF()
  ENDIF()
ENDMACRO()


FUNCTION(PACKAGE_ARCH_SETUP_MPI)

  #
  # A) Get the directory paths
  #
  
  MULTILINE_SET( DOC
    "Base directory for the MPI implementation under which"
    " the bin, include, and lib directories are found" )
  ADVANCED_SET( MPI_BASE_DIR "" CACHE PATH ${DOC} )
  PRINT_NONEMPTY_VAR(MPI_BASE_DIR)

  IF (MPI_BASE_DIR)
    SET(MPI_BIN_DIR_DEFAULT "${MPI_BASE_DIR}/bin")
  ELSE()
    SET(MPI_BIN_DIR_DEFAULT "")
  ENDIF()
  MULTILINE_SET( DOC
    "Path to the bin directory where the MPI compiler"
    " and runtime executables are found" )
  ADVANCED_SET( MPI_BIN_DIR ${MPI_BIN_DIR_DEFAULT} CACHE PATH ${DOC} )
  PRINT_NONEMPTY_VAR(MPI_BIN_DIR)

  MULTILINE_SET( DOC
    "If set to 'ON', then the MPI compiler wrappers will be used."
    "  Set MPI_[C,CXX,Fortran]_COMPILER:FILEPATH=XXX to set compilers." )
  ADVANCED_SET( MPI_USE_COMPILER_WRAPPERS ON CACHE BOOL ${DOC} )
  PRINT_VAR(MPI_USE_COMPILER_WRAPPERS)

  FILE(TO_CMAKE_PATH "$ENV{ProgramFiles}" PROGRAM_FILES)
  IF(MPI_BIN_DIR)
    SET(MPI_BIN_DIR_PATHS ${MPI_BIN_DIR})
  ELSE()
    SET(MPI_BIN_DIR_PATHS
      /usr/local/mpi/bin
      /usr/local/bin
      /usr/bin
      "${PROGRAM_FILES}/Microsoft HPC Pack 2008 SDK/Bin"
      "C:/Program Files/Microsoft HPC Pack 2008 SDK/Bin"
      "${PROGRAM_FILES}/MPICH/SDK/Bin"
      "${PROGRAM_FILES}/MPICH2/Bin"
      "C:/Program Files/MPICH/SDK/Bin"
      "C:/Program Files/MPICH2/Bin"
      )
  ENDIF()

  #
  # B) Get the MPI compilers and/or just the raw include paths and libraries
  #

  IF (MPI_USE_COMPILER_WRAPPERS)

    # B.1) Set up to use the MPI wrappers

    PACKAGE_ARCH_FIND_MPI_COMPILER(C mpicc)

    PACKAGE_ARCH_FIND_MPI_COMPILER(CXX  mpicxx mpic++ mpiCC)

    PACKAGE_ARCH_FIND_MPI_COMPILER(Fortran mpif90 mpif77)

  ELSE()

    # B.2) Set up to use raw configure options

    ADVANCED_SET( MPI_COMPILE_FLAGS ""
      CACHE STRING
      "List of general compiler flags (excluding include directories)." )

    ADVANCED_SET( MPI_LINK_FLAGS ""
      CACHE STRING
      "Link Flags for MPI executables." )

    # NOTE: Test rest of the flags will be set up by the
    # FindTPLMPI.cmake module!

  ENDIF()

  #
  # C) Get the MPI executable
  #

   FIND_PROGRAM_PLUS( MPI_EXEC
    NAMES mpiexec mpirun
    PATHS ${MPI_BIN_DIR_PATHS}
    DOC "MPI executable used to run MPI programs"
    )
  MARK_AS_ADVANCED(MPI_EXEC)

  IF(MPI_EXEC)

    GET_FILENAME_COMPONENT( MPI_EXEC_NAME "${MPI_EXEC}" PATH )

    IF(MPI_EXEC_NAME STREQUAL mpiexec)
      SET(MPI_EXEC_NUMPROCS_FLAG_DEFAULT -n)
    ELSE()
      SET(MPI_EXEC_NUMPROCS_FLAG_DEFAULT -np)
    ENDIF()
    ADVANCED_SET( MPI_EXEC_NUMPROCS_FLAG
      ${MPI_EXEC_NUMPROCS_FLAG_DEFAULT}
      CACHE STRING
      "Flag setting the number of processors to use with MPI run command." )
    IF (${PROJECT_NAME}_VERBOSE_CONFIGURE)
      PRINT_NONEMPTY_VAR(MPI_EXEC_NUMPROCS_FLAG)
    ENDIF()

    ADVANCED_SET( MPI_EXEC_MAX_NUMPROCS "4"
      CACHE STRING
      "The maximum mumber of processes to use when running MPI programs." )

    ADVANCED_SET( MPI_EXEC_PRE_NUMPROCS_FLAGS ""
      CACHE STRING
      "Extra command-line args to the MPI exec before num-procs args." )

    ADVANCED_SET( MPI_EXEC_POST_NUMPROCS_FLAGS ""
      CACHE STRING
      "Extra command-line args to the MPI exec after num-procs args." )

  ENDIF()

  #MESSAGE(FATAL_ERROR "Stopping!")

ENDFUNCTION()

# 2009/01/23: rabartl: ToDo: Above: create util FIND_PROGRAM_PATH_FIRST(...) 
# in order to implement looking in the input path first and not last
