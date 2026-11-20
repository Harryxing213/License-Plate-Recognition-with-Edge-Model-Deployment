#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rknn_api.h"

static void* read_file(const char* path, size_t* out_size) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (size <= 0) { fclose(f); return NULL; }
  void* data = malloc((size_t)size);
  if (!data) { fclose(f); return NULL; }
  if (fread(data, 1, (size_t)size, f) != (size_t)size) { free(data); fclose(f); return NULL; }
  fclose(f);
  *out_size = (size_t)size;
  return data;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s model.rknn input.raw [loops]\n", argv[0]);
    return 1;
  }
  int loops = argc > 3 ? atoi(argv[3]) : 100;
  if (loops <= 0) loops = 100;

  size_t model_size = 0;
  void* model_data = read_file(argv[1], &model_size);
  if (!model_data) return 2;
  rknn_context ctx = 0;
  int ret = rknn_init(&ctx, model_data, (uint32_t)model_size, 0, NULL);
  free(model_data);
  if (ret != RKNN_SUCC) {
    printf("rknn_init ret=%d\n", ret);
    return 3;
  }

  rknn_tensor_attr in_attr;
  memset(&in_attr, 0, sizeof(in_attr));
  in_attr.index = 0;
  rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));

  rknn_tensor_attr out_attr;
  memset(&out_attr, 0, sizeof(out_attr));
  out_attr.index = 0;
  rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));

  uint32_t input_size = in_attr.size_with_stride ? in_attr.size_with_stride : in_attr.size;
  uint32_t output_size = out_attr.size_with_stride ? out_attr.size_with_stride : out_attr.size;
  rknn_tensor_mem* input_mem = rknn_create_mem(ctx, input_size);
  rknn_tensor_mem* output_mem = rknn_create_mem(ctx, output_size);
  if (!input_mem || !output_mem) return 4;

  size_t raw_size = 0;
  void* raw = read_file(argv[2], &raw_size);
  if (!raw) return 5;
  memset(input_mem->virt_addr, in_attr.zp, input_size);
  memcpy(input_mem->virt_addr, raw, raw_size < input_size ? raw_size : input_size);
  free(raw);
  memset(output_mem->virt_addr, 0, output_size);

  in_attr.pass_through = 1;
  out_attr.pass_through = 0;
  ret = rknn_set_io_mem(ctx, input_mem, &in_attr);
  if (ret != RKNN_SUCC) return 6;
  ret = rknn_set_io_mem(ctx, output_mem, &out_attr);
  if (ret != RKNN_SUCC) return 7;

  rknn_mem_sync(ctx, input_mem, RKNN_MEMORY_SYNC_TO_DEVICE);
  for (int i = 0; i < 10; ++i) rknn_run(ctx, NULL);

  double min_ms = 1e9;
  double max_ms = 0.0;
  double sum_ms = 0.0;
  for (int i = 0; i < loops; ++i) {
    double t0 = now_ms();
    ret = rknn_run(ctx, NULL);
    double t1 = now_ms();
    if (ret != RKNN_SUCC) return 8;
    double dt = t1 - t0;
    if (dt < min_ms) min_ms = dt;
    if (dt > max_ms) max_ms = dt;
    sum_ms += dt;
  }

  printf("loops=%d\n", loops);
  printf("run_only_ms avg=%.3f min=%.3f max=%.3f\n", sum_ms / loops, min_ms, max_ms);

  rknn_destroy_mem(ctx, input_mem);
  rknn_destroy_mem(ctx, output_mem);
  rknn_destroy(ctx);
  return 0;
}
