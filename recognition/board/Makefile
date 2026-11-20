# Use the compiler, headers and librknnmrt from the RV1106 SDK.
CC = arm-rockchip830-linux-uclibcgnueabihf-gcc
RKNN_INCLUDE ?=
RKNN_LIB ?=
CFLAGS ?= -O2 -Wall
LDLIBS = -lrknnmrt
TARGETS = bin/rknn_smoke_zero_copy_logical_output bin/rknn_benchmark_run_only

all: $(TARGETS)

bin/%: %.c
	@test -n "$(RKNN_INCLUDE)" -a -n "$(RKNN_LIB)" || (echo "Set RKNN_INCLUDE and RKNN_LIB from your RV1106 SDK"; exit 1)
	mkdir -p bin
	$(CC) $(CFLAGS) -I"$(RKNN_INCLUDE)" $< -L"$(RKNN_LIB)" $(LDLIBS) -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
