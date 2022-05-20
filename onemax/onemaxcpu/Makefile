CC = g++
NVCC = nvcc
gcc_options = -std=c++17 -Wall --pedantic-error
INCLUDE = -I./include

#program : population.cpp population.hpp\
#	  individual.cpp individual.hpp\
#	  floyd.cpp\
#	  all.h all.h.gch
#	g++ %(gcc_options) -include all.h $< -o $@

PROGRAM = onemax

all : $(PROGRAM)

# $@ file name of target
# $< file name of predefined
#all.h.gch : all.h
#	g++ $(gcc_options) -x c++-header -o $@ $<

individual.o : individual.cpp individual.hpp #all.h.gch
	$(CC) $(gcc_options) ${INCLUDE} -c $< -o $@
	#$(CC) $(gcc_options) -include all.h -c $< -o $@
	#$(CC) $(gcc_options) -include all.h $< -c individual.cpp

population.o : population.cpp population.hpp #all.h.gch
	$(CC) $(gcc_options) ${INCLUDE} -c $< -o $@
	#$(CC) $(gcc_options) -include all.h -c $< -o $@
	# $(CC) $(gcc_options) -include all.h $< -c population.cpp

parameters.o : parameters.cpp parameters.hpp
	$(CC) $(gcc_options) ${INCLUDE} -c $< -o $@
	#$(CC) $(gcc_options) -include all.h -c $< -o $@

onemax : onemax.cpp individual.o population.o parameters.o #all.h.gch
	$(CC) $(gcc_options) ${INCLUDE} $^ -o onemax
	#$(CC) $(gcc_options) -include all.h $^ -o onemax

clean :
	rm -f ${PROGRAM}
	rm -f *.o




