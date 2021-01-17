CC				:= g++
TARGET		:= dist/matrix
BUILDDIR	:= build
SRCDIR		:= src
CFLAGS		:= -std=c++17 -g -pthread -mavx2
SRCEXT		:= cpp
SOURCES		:= $(shell find $(SRCDIR) -type f -name '*.$(SRCEXT)')
OBJECTS		:= $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(SOURCES:.$(SRCEXT)=.o))

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@printf "\e[33m\e[1mBuilding...\e[0m\n";
	@mkdir -p $(BUILDDIR)
	@echo "  $(notdir $@) from $(notdir $<)"
	@$(CC) $(CFLAGS) -c -o $@ $<

$(TARGET): $(OBJECTS)
	@printf "\e[95m\e[1mLinking...\e[0m\n";
	@echo "  $(notdir $(OBJECTS))"
	@mkdir -p dist
	@$(CC) $(CFLAGS) -o $@ $^ $(LIB)
	@rm -r $(BUILDDIR)

PHONY: run
run:
	@mkdir -p $(BUILDDIR)
	@for source in $(basename $(notdir $(SOURCES))); do\
		printf "\e[33m\e[1mBuilding...\e[0m\n";\
		echo "  $$source.o from $$source.$(SRCEXT)";\
		$(CC) $(CFLAGS) -c -o $(BUILDDIR)/$$source.o $(SRCDIR)/$$source.$(SRCEXT);\
	done
	@printf "\e[95m\e[1mLinking...\e[0m\n";
	@echo "  $(notdir $(OBJECTS))"
	@mkdir -p dist
	@$(CC) $(CFLAGS) $(LIB) -o $(TARGET) $(OBJECTS)
	@rm -r $(BUILDDIR)
	@printf "\e[32m\e[1mRunning $(notdir $(TARGET))...\e[0m\n"
	@./$(TARGET)

PHONY: clean
clean:
	@printf "\e[91m\e[1mCleaning...\e[0m\n"
	@echo "  /$(BUILDDIR)"
	@echo "  /$(TARGET)"
	@$(RM) -r $(BUILDDIR) $(OBJECTS)
	@$(RM) "./$(TARGET)"

PHONY: r
r:
	@printf "\e[32m\e[1mRunning $(notdir $(TARGET))...\e[0m\n"
	@./$(TARGET)