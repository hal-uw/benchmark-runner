ROCM_DIR ?= /opt/rocm-7.1.0
CXX ?= g++
CXXFLAGS ?= -O2 -I$(ROCM_DIR)/include
LDFLAGS ?= -L$(ROCM_DIR)/lib -lamd_smi

TARGET := amd-smi-query
SRC := power_query.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean
