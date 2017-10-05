#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

void caffe_test_dumpBuffer_relu(void* buf, size_t numElements, std::string layername, std::string path)
{
    // Replace '/' to '_' in the layername
    std::string fileName = layername;
    size_t start_pos = 0;
    while ((start_pos = fileName.find("/", start_pos)) != std::string::npos) {
        fileName.replace(start_pos, 1, "_");
        start_pos += 1; // Handles case where 'to' is a substring of 'from'
    }
    fileName = path + fileName + ".f32";
    printf("CAFFE RELU WRITE: Writing file %s with %d elements\n", fileName.c_str(), (int)numElements);

    FILE * fp = fopen(fileName.c_str(), "wb");
    if(!fp) printf("Could not open file %s\n", fileName.c_str());
    else
    {
        printf("CAFFE RELU WRITE: Writing file %s into caffeBufferDump folder\n", fileName.c_str());
        fwrite(buf, sizeof(float), numElements, fp);
    }
    fclose(fp);
}

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
/*DUMP LAYER BUFFER*/
#if CAFFE_BUFFER_DUMP
#if _WIN32
  CreateDirectory("caffeBufferDump", NULL);
#else
  struct stat st = {0};
if (stat("caffeBufferDump", &st) == -1) { mkdir("caffeBufferDump", 0700); }
#endif
//if(this->layer_param().name() == "relu1_1")
  caffe_test_dumpBuffer(top[0]->mutable_cpu_data(), top[0]->count(), this->layer_param().name(), "caffeBufferDump/");
#endif
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
