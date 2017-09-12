#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include<stdio.h>

void caffe_test_dumpBuffer_conv(void* buf, size_t numElements, std::string layername, std::string path)
{
    // Replace '/' to '_' in the layername
    std::string fileName = layername;
    size_t start_pos = 0;
    while ((start_pos = fileName.find("/", start_pos)) != std::string::npos) {
        fileName.replace(start_pos, 1, "_");
        start_pos += 1; // Handles case where 'to' is a substring of 'from'
    }
    fileName = path + fileName + ".f32";
    printf("CAFFE CONV WRITE:: Writing file %s with %d elements\n", fileName.c_str(), (int)numElements);

    FILE * fp = fopen(fileName.c_str(), "wb");
    if(!fp) printf("Could not open file %s\n", fileName.c_str());
    else
    {
        printf("CAFFE CONV WRITE:: Writing file %s into caffe_local_output folder\n", fileName.c_str());
        fwrite(buf, sizeof(float), numElements, fp);
    }
    fclose(fp);
}

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
/*DUMP LAYER BUFFER*/
#if CAFFE_BUFFER_DUMP
#if _WIN32
  CreateDirectory("caffe_local_output", NULL);
#else
  struct stat st = {0};
if (stat("caffe_local_output", &st) == -1) { mkdir("caffe_local_output", 0700); }
#endif
  //if(this->layer_param().name() == "conv1_1")
    caffe_test_dumpBuffer_conv(top[0]->mutable_cpu_data(), top[0]->count(), this->layer_param().name(), "caffe_local_output/");
#endif
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
