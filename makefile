CXX      = g++
CXXFLAGS = -O3 -fopenmp -std=c++17 \
           $(shell pkg-config --cflags opencv4)
LDFLAGS  = $(shell pkg-config --libs opencv4)

SRCS = v1/sobel_v1.cpp v2/sobel_v2.cpp v3/sobel_v3.cpp
OBJS = $(SRCS:.cpp=.o)

all: sobel_main sobel_bench

sobel_main: main.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

sobel_bench: benchmark.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

v1/%.o: v1/%.cpp v1/%.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

v2/%.o: v2/%.cpp v2/%.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

v3/%.o: v3/%.cpp v3/%.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f v1/*.o v2/*.o v3/*.o sobel_main sobel_bench

.PHONY: all clean