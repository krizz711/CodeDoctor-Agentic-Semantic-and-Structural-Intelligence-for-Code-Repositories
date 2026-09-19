CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude

ifeq ($(OS),Windows_NT)
MKDIR_BUILD := if not exist build mkdir build
REMOVE_BUILD := if exist build rmdir /s /q build
else
MKDIR_BUILD := mkdir -p build
REMOVE_BUILD := rm -rf build
endif

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
	@$(MKDIR_BUILD)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

$(PRODUCT_TARGET): $(PRODUCT_SRC)
	@$(MKDIR_BUILD)
	$(CXX) $(CXXFLAGS) $(PRODUCT_SRC) -o $(PRODUCT_TARGET)

$(TEST_TARGET): $(CORE_SRC) tests/cpp_core_test.cpp
	@$(MKDIR_BUILD)
	$(CXX) $(CXXFLAGS) $(CORE_SRC) tests/cpp_core_test.cpp -o $(TEST_TARGET)

clean:
	@$(REMOVE_BUILD)

.PHONY: all clean
