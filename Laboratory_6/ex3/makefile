CC=nvcc
CXXFLAGS=-O3 -g
LIBS=
INCLUDES=-I../../
LIB_FLAGS=-lm

MAIN_NAME=cuda_element_add
MAIN_BIN=$(MAIN_NAME)
MAIN_SRC=$(MAIN_NAME).cu

all: $(MAIN_BIN)

$(MAIN_BIN): $(MAIN_SRC)
	$(CC) $(CXXFLAGS) $^ -o $(MAIN_BIN) $(LIBS) $(INCLUDES) $(LIB_FLAGS)

clean:
	rm -rf $(MAIN_BIN)
