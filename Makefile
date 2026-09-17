CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude
TARGET := build/codedoctor_demo.exe
PRODUCT_TARGET := build/codedoctor.exe
TEST_TARGET := build/codedoctor_cpp_test.exe
SRC := src/chunker.cpp src/indexer.cpp src/retriever.cpp src/graph.cpp demo_repo/src/demo.cpp
CORE_SRC := src/chunker.cpp src/indexer.cpp src/retriever.cpp src/graph.cpp
PRODUCT_SRC := src/chunker.cpp src/indexer.cpp src/retriever.cpp src/graph.cpp src/repository.cpp src/main.cpp

all: $(PRODUCT_TARGET) $(TARGET)

test: $(TEST_TARGET)
	$(TEST_TARGET)

$(TARGET): $(SRC)
	@if not exist build mkdir build
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

$(PRODUCT_TARGET): $(PRODUCT_SRC)
	@if not exist build mkdir build
	$(CXX) $(CXXFLAGS) $(PRODUCT_SRC) -o $(PRODUCT_TARGET)

$(TEST_TARGET): $(CORE_SRC) tests/cpp_core_test.cpp
	@if not exist build mkdir build
	$(CXX) $(CXXFLAGS) $(CORE_SRC) tests/cpp_core_test.cpp -o $(TEST_TARGET)

clean:
	@if exist build rmdir /s /q build

.PHONY: all clean
