#ifndef SPMM_OPT_H_
#define SPMM_OPT_H_

#include "spmm_base.h"

struct Task {
  int row;
  int ptr_begin;
  int ptr_end;
};

class SpMMOpt : public SpMM {
 public:
  SpMMOpt(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e,
          int out_feat_in)
      : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
  SpMMOpt(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
  ~SpMMOpt() {
    if (this->d_tasks_) checkCudaErrors(cudaFree(this->d_tasks_));
  }
  virtual void preprocess(float *vin, float *vout);
  virtual void run(float *vin, float *vout);

 private:
  int num_tasks_;
  Task *d_tasks_;
  int num_nonzero_rows;
  int *d_row_reorder_;
};

#endif
