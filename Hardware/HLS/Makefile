DEVICE     := 5CSEBA6U23I7
ECHO       := echo
TARGETS    := clean csim csynth
COMP_FILE  := top.cpp
TB_FILE    := top_tb.cpp 
			
CXX    := i++
RM     := rm -rf

CURRENT_DIR = $(shell pwd)
PARAMS_PATH = $(CURRENT_DIR)/../../Training/export/
UTILS_PATH = $(CURRENT_DIR)/utils/
HLSLIB_PATH = $(CURRENT_DIR)/ornn-hlslib
CXXFLAGS := -I$(PARAMS_PATH) -I$(UTILS_PATH) -I$(HLSLIB_PATH)

.PHONY: all
all: $(TARGETS)

clean:
	-$(RM) csim *.prj *tmp

csim: $(TB_FILE) clean
	@$(ECHO) "Running CSIM..."
	$(CXX) $(CXXFLAGS) -v -march=x86-64 $< -o $@
	./$@

csynth: $(COMP_FILE)
	@$(ECHO) "Running CSYNTH..."
	$(CXX) $(CXXFLAGS) -v --simulator none --quartus-compile -march=$(DEVICE) $< -o $@

