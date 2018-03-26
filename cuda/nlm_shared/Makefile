SHELL := /bin/bash # Use bash syntax

CC = sm_20

NVCC	= nvcc -arch=$(CC)
nlm_shared	= nlmSharedKernel
ARGS	= -ptx

all: nlm_shared

nlm_shared:
	$(NVCC) $(ARGS) $(nlm_shared).cu

clean:
	rm -rf *~ *.ptx
