MAKEFLAGS += -r

CC = gcc

INCLUDE_DIR = ./includes/
INCLUDE_SUBDIRS = $(INCLUDE_DIR) $(INCLUDE_DIR)activations/ $(INCLUDE_DIR)matrices/
OUT_DIR = ./bin/
INCLUDE_OUT_DIR = ./bin/includes/
INCLUDE_OUT_SUBDIRS = $(subst $(INCLUDE_DIR),$(INCLUDE_OUT_DIR),$(INCLUDE_SUBDIRS))
TARGET = $(OUT_DIR)main.exe
MAIN = main.c

SRC = $(foreach dir,$(INCLUDE_SUBDIRS),$(wildcard $(dir)*.c))
OBJECTS = $(subst $(INCLUDE_DIR),$(INCLUDE_OUT_DIR),$(SRC:.c=.o))

.PHONY: debug all

debug:
	@echo SRC = $(SRC)
	@echo OBJECTS = $(dir $(OBJECTS))
	@echo $(INCLUDE_OUT_SUBDIRS)

all: $(TARGET)

$(INCLUDE_OUT_SUBDIRS):
	mkdir -p $(OUT_DIR) 
	mkdir -p $(dir $(INCLUDE_OUT_SUBDIRS))

$(INCLUDE_OUT_DIR)%.o: $(INCLUDE_DIR)%.c | $(INCLUDE_OUT_SUBDIRS)
	$(CC) -x c \
		-I$(INCLUDE_DIR) \
		-o $@ \
		-c $^ \
		-D$(shell echo $(patsubst %.c,%,$(lastword $(subst /, ,$^)))_IMPLEMENTATION | tr [:lower:] [:upper:]) \
		-lm

$(TARGET): $(MAIN) $(OBJECTS)
	$(CC) -o $(TARGET) $(OBJECTS) $(MAIN) -I$(INCLUDE_DIR)

clean:
	rm -rf ./bin/*

