CC = g++
NVCC = nvcc
gcc_options = -std=c++17 -Wall --pedantic-error
#gcc_options = -std=c++17 -Wall --pedantic-error -D_DEBUG
NVCC_OPTIONS = -arch=sm_80

OPTIONS =
#OPTIONS += -D_ELAPSED_TIME
#OPTIONS += -D_DEBUG_FITNESS
#OPTIONS += -D_DEBUG_ELITE
#OPTIONS += -D_DEBUG_CHROMOSOME
OPTIONS += -D_TREND

INCLUDE = -I./include

#program : population.cpp population.hpp\
#	  individual.cpp individual.hpp\ floyd.cpp\
#	  all.h all.h.gch
#	g++ %(gcc_options) -include all.h $< -o $@

PROGRAM = onemax

all : $(PROGRAM)

# $@ file name of target
# $< file name of predefined
#all.h.gch : all.h
#	g++ $(gcc_options) -x c++-header -o $@ $<

individual.o : individual.cpp individual.hpp #all.h.gch
	$(CC) $(gcc_options) ${OPTIONS} ${INCLUDE} -c $< -o $@
	#$(CC) $(gcc_options) -include all.h -c $< -o $@
	#$(CC) $(gcc_options) -include all.h $< -c individual.cpp

population.o : population.cpp population.hpp #all.h.gch
	# $(CC) $(gcc_options) ${INCLUDE} -c $< -o $@
	$(CC) $(gcc_options) ${OPTIONS} ${INCLUDE} -c $< -o $@
	#$(CC) $(gcc_options) -include all.h -c $< -o $@
	#$(CC) $(gcc_options) -include all.h $< -c population.cpp

parameters.o : parameters.cpp parameters.hpp
	#$(CC) $(gcc_options) ${INCLUDE} -c $< -o $@
	$(CC) $(gcc_options) ${OPTIONS} ${INCLUDE} -c $< -o $@
	#$(CC) $(gcc_options) -include all.h -c $< -o $@

onemax : onemax.cu individual.o population.o parameters.o #all.h.gch
	$(NVCC) ${NVCC_OPTIONS} ${OPTIONS} ${INCLUDE} $^ -o onemax
	#$(CC) $(gcc_options) ${OPTIONS} ${INCLUDE} $^ -o onemax
	#$(CC) $(gcc_options) ${INCLUDE} $^ -o onemax
	#$(CC) $(gcc_options) -include all.h $^ -o onemax

clean :
	rm -f ${PROGRAM}
	rm -f *.o




