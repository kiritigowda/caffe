// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

void caffe_test_dumpBuffer_dropout(void* buf, size_t numElements, std::string layername, std::string path)
{
    // Replace '/' to '_' in the layername
    std::string fileName = layername;
    size_t start_pos = 0;
    while ((start_pos = fileName.find("/", start_pos)) != std::string::npos) {
        fileName.replace(start_pos, 1, "_");
        start_pos += 1; // Handles case where 'to' is a substring of 'from'
    }
    fileName = path + fileName + ".f32";
    printf("CAFFE DROPOUT WRITE: Writing file %s with %d elements\n", fileName.c_str(), (int)numElements);

    FILE * fp = fopen(fileName.c_str(), "wb");
    if(!fp) printf("Could not open file %s\n", fileName.c_str());
    else
    {
        printf("CAFFE DROPOUT WRITE: Writing file %s into caffe_local_output folder\n", fileName.c_str());
        fwrite(buf, sizeof(float), numElements, fp);
    }
    fclose(fp);
}

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
/*DUMP LAYER BUFFER*/
#if CAFFE_BUFFER_DUMP
#if _WIN32
  CreateDirectory("caffe_local_output", NULL);
#else
  struct stat st = {0};
if (stat("caffe_local_output", &st) == -1) { mkdir("caffe_local_output", 0700); }
#endif
//if(this->layer_param().name() == "drop6")
  caffe_test_dumpBuffer_dropout(top[0]->mutable_cpu_data(), top[0]->count(), this->layer_param().name(), "caffe_local_output/");
#endif
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS(DropoutLayer);
REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
