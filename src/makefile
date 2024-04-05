# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects

.PHONY = all clean

CC = gfortran		

SRCS1  = load_options.f90 main.f90
SRCS  := $(filter-out ${SRCS1},$(wildcard *.f90)) 
OBJS  := $(SRCS:%.f90=%.o)
LIB   := libcosmo.a 

all: ${OBJS} makelib 

# compiling each source file in src 
%.o: %.f90
	@echo "Creating object $@ from $< ..."
	${CC} -c $< -o $@

# creating the main library
makelib:  
	@echo "Creating libary..."
	ar rcs ${LIB} ${OBJS}
	@echo "Cleaning up ..."
	rm -rvf *.o ${OBJS}

# building the main app
build:
	@echo "Building app..."
	${CC} ${SRCS1} ${LIB} -o cosmocalc

# clean all
clean:
	@echo "Cleaning up ..."
	rm -rvf *.o ${OBJS}