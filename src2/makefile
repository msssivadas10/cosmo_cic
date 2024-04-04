# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects

.PHONY = all clean

# compiler to use
CC = gfortran		

# files to exclude while compiling
EXCL = ztest_001.f90 ztest_002.f90 main.f90

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


# clean all .o files
cleanobjs:
	@echo "Cleaning up ..."
	rm -rvf *.o ${OBJS}

# clean all
clean:
	@echo "Cleaning up ..."
	rm -rvf *.o ${OBJS}