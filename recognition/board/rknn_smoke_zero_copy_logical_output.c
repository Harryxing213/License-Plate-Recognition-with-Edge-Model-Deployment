#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static const char* label_name(int idx) {
  static const char* names[] = {
      "<PAD>", "<BOS>", "<EOS>", "wan", "hu", "jin", "yu", "ji", "jin2", "meng", "liao", "ji2", "hei", "su", "zhe", "jing", "min", "gan", "lu", "yu2",
      "e", "xiang", "yue", "gui", "qiong", "chuan", "gui2", "yun", "zang", "shan", "gan2", "qing", "ning", "xin", "jing2", "xue",
      "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
      "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O"};
  if (idx < 0 || idx >= (int)(sizeof(names) / sizeof(names[0]))) return "?";
  return names[idx];
}

int main(int argc, char** argv) {
  if (argc < 3) return 1;
  size_t model_size = 0;
  void* model_data = read_file(argv[1], &model_size);
  if (!model_data) return 2;
  rknn_context ctx = 0;
  int ret = rknn_init(&ctx, model_data, (uint32_t)model_size, 0, NULL);
  free(model_data);
  printf("rknn_init ret=%d\n", ret);
  if (ret != RKNN_SUCC) return 3;

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
  memset(input_mem->virt_addr, in_attr.zp, input_size);
  memset(output_mem->virt_addr, 0, output_size);

  size_t raw_size = 0;
  void* raw = read_file(argv[2], &raw_size);
  if (!raw) return 5;
  size_t copy_size = raw_size < input_size ? raw_size : input_size;
  memcpy(input_mem->virt_addr, raw, copy_size);
  free(raw);
  printf("loaded input=%s bytes=%zu copied=%zu\n", argv[2], raw_size, copy_size);

  in_attr.pass_through = 1;
  out_attr.pass_through = 0;
  ret = rknn_set_io_mem(ctx, input_mem, &in_attr);
  printf("rknn_set_io_mem input ret=%d\n", ret);
  if (ret != RKNN_SUCC) return 6;
  ret = rknn_set_io_mem(ctx, output_mem, &out_attr);
  printf("rknn_set_io_mem output ret=%d\n", ret);
  if (ret != RKNN_SUCC) return 7;

  printf("input sync ret=%d\n", rknn_mem_sync(ctx, input_mem, RKNN_MEMORY_SYNC_TO_DEVICE));
  ret = rknn_run(ctx, NULL);
  printf("rknn_run ret=%d\n", ret);
  if (ret != RKNN_SUCC) return 8;
  printf("output sync ret=%d\n", rknn_mem_sync(ctx, output_mem, RKNN_MEMORY_SYNC_FROM_DEVICE));

  int8_t* out = (int8_t*)output_mem->virt_addr;
  printf("output first_values=");
  for (uint32_t i = 0; i < out_attr.n_elems && i < 8; ++i) {
    float v = ((float)out[i] - (float)out_attr.zp) * out_attr.scale;
    printf("%s%.6f", i ? "," : "", v);
  }
  printf("\n");

  int preds[8];
  printf("pred_indices=");
  for (int pos = 0; pos < 8; ++pos) {
    int best = 0;
    int8_t best_v = out[pos * 70];
    for (int cls = 1; cls < 70; ++cls) {
      int8_t v = out[pos * 70 + cls];
      if (v > best_v) { best_v = v; best = cls; }
    }
    preds[pos] = best;
    printf("%s%d", pos ? "," : "", best);
  }
  printf("\npred_labels=");
  for (int pos = 0; pos < 8; ++pos) printf("%s%s", pos ? "," : "", label_name(preds[pos]));
  printf("\n");

  rknn_destroy_mem(ctx, input_mem);
  rknn_destroy_mem(ctx, output_mem);
  rknn_destroy(ctx);
  return 0;
}
