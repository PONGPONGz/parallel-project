Compile: mpicc -g -I/opt/homebrew/Cellar/open-mpi/5.0.5/include -o main main.c  
Run: mpirun -np <number of compute nodes> ./main
