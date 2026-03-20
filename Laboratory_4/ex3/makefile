CC=gcc

LIBS=-L/opt/shares/openfoam/software/OpenBLAS/0.3.27-GCC-13.3.0/lib								# find libraries
INCLUDES=-I/opt/shares/openfoam/software/OpenBLAS/0.3.27-GCC-13.3.0/include/ -Iinclude			# find header files
LIB_FLAGS=-lm -lopenblas		# link to the libraries

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BATCH_OUT_FOLDER := outputs


BLOCK_NAME=block_gemm
BLOCK_BIN=$(BLOCK_NAME)
BLOCK_SRC=$(BLOCK_NAME).c

OBJECTS = $(OBJ_FOLDER)/my_time_lib.o

all: $(BIN_FOLDER)/$(BLOCK_NAME)

$(OBJ_FOLDER)/my_time_lib.o: $(SRC_FOLDER)/my_time_lib.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/my_time_lib.c -o $@ $(LIB_FLAGS)


$(BIN_FOLDER)/$(BLOCK_NAME): $(BLOCK_SRC) $(OBJECTS)
	mkdir -p $(BIN_FOLDER)
	$(CC) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)


clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)
