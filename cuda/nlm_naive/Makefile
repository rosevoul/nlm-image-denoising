SHELL := /bin/bash # Use bash syntax

CC = sm_20

NVCC	= nvcc -arch=$(CC)
nlm_naive	= nlmNaiveKernel
ARGS	= -ptx

all: nlm_naive

nlm_naive:
	$(NVCC) $(ARGS) $(nlm_naive).cu

clean:
	rm -rf *~ *.ptx
