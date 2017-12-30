###########################################################

# USER SPECIFIC DIRECTORIES

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

# NVCC COMPILER OPTIONS

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= -gencode arch=compute_30,code=sm_30
NVCC_LIBS=

# NVCC library directory:
NVCC_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# NVCC include directory:
NVCC_INC_DIR= -I$(CUDA_ROOT_DIR)/include

##########################################################

# Make variables:

# Executable name:
EXE = test_find_value

##########################################################

# Compile:

# Compile all objects:
all: $(EXE)

# Compile to executable:
$(EXE):Find_Value_CUDA.cu
	$(NVCC) -o $(EXE) Find_Value_CUDA.cu

# Clean objects in object directory.
clean:
	$(RM) $(OBJ_DIR)/*.cu_o $(EXE) $(EXE).*




