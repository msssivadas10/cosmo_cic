# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects

.PHONY = all clean

# compiler to use
CC = gfortran		

# program file
PROG = test.f90 # main program file
# PROG = test.f90 # for testing

# files to exclude while compiling
EXCL = test.f90 

# selecting all source files in src, except main files
SRCS := $(filter-out ${EXCL},$(wildcard *.f90))
OBJS := $(SRCS:%.f90=%.o)

# library
LIB := libcosmo.a

all: ${OBJS} makelib cleanobjs

# compiling each source file in src 
%.o: %.f90
	@echo "Creating object $@ from $< ..."
	${CC} -c $< -o $@

# creating the main library
makelib:  
	@echo "Creating libary..."
	ar rcs ${LIB} ${OBJS}
#	$(CC) ${OBJS} $(PROG) src/chebyshev.f90 -o $@

# compite main
#install:
#	@echo "Compiling sources"
#	$(CC) $(PROG) ${LIB}

# clean all .o files
cleanobjs:
	@echo "Cleaning up ..."
	rm -rvf *.o ${OBJS}

# clean all
clean:
	@echo "Cleaning up ..."
	rm -rvf *.o ${OBJS}