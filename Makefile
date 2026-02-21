MAKEFLAGS += -r

CC = gcc
FLAGS = -Wall

INCLUDE_DIR = ./includes/
INCLUDE_SUBDIRS = $(INCLUDE_DIR) $(INCLUDE_DIR)activations/ $(INCLUDE_DIR)matrices/
OUT_DIR = ./bin/
INCLUDE_OUT_DIR = ./bin/includes/
INCLUDE_OUT_SUBDIRS = $(subst $(INCLUDE_DIR),$(INCLUDE_OUT_DIR),$(INCLUDE_SUBDIRS))
TARGET = $(OUT_DIR)main.exe
MAIN = main.c

SRC = $(foreach dir,$(INCLUDE_SUBDIRS),$(wildcard $(dir)*.c))
IMPL_SRC = $(foreach dir,$(INCLUDE_SUBDIRS),$(wildcard $(dir)*.inc))
HEADER_SRC = $(foreach dir,$(INCLUDE_SUBDIRS),$(wildcard $(dir)*.h))

DEPENDENCIES = $(IMPL_SRC) $(HEADER_SRC)

OBJECTS = $(subst $(INCLUDE_DIR),$(INCLUDE_OUT_DIR),$(SRC:.c=.o))

.PHONY: debug all ctags

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
		$(FLAGS) \
		-lm

$(TARGET): $(MAIN) $(OBJECTS) $(DEPENDENCIES)
	$(CC) -o $(TARGET) $(OBJECTS) $(MAIN) -I$(INCLUDE_DIR) $(FLAGS)

clean:
	rm -rf ./bin/*

debug:
	@echo $(DEPENDENCIES)

ctags:
	ctags -R --kinds-C=+defghlmpstuvxzLD .


