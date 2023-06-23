#ifndef COVARIANCE_HPP
#define COVARIANCE_HPP

#define _USE_MATH_DEFINES

#include "openmpheader.h"
#include "general.h"
#include "interpreter.h"
#include "formula.hpp"
#include "sparse.h"

using namespace Eigen;

namespace glmmr {

class Covariance {
public:
  glmmr::Formula form_;
  const ArrayXXd data_;
  const strvec colnames_;
  dblvec parameters_;
  dblvec other_pars_;
  
  Covariance(const str& formula,
             const ArrayXXd &data,
             const strvec& colnames) :
    form_(formula), data_(data), colnames_(colnames), Q_(0),
    size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()) { 
      Z_constructor();
    };

  Covariance(const glmmr::Formula& form,
             const ArrayXXd &data,
             const strvec& colnames) :
    form_(form), data_(data), colnames_(colnames), Q_(0),
    size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()) {
      Z_constructor();
    };

  Covariance(const str& formula,
             const ArrayXXd &data,
             const strvec& colnames,
             const dblvec& parameters) :
    form_(formula), data_(data), colnames_(colnames), parameters_(parameters),
    Q_(0),size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()), spchol((make_sparse(),mat)) {
      L_constructor();
      Z_constructor();
    };

  Covariance(const glmmr::Formula& form,
             const ArrayXXd &data,
             const strvec& colnames,
             const dblvec& parameters) :
    form_(form), data_(data), colnames_(colnames), parameters_(parameters),
    Q_(0),size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()), spchol((make_sparse(),mat)) {
      L_constructor();
      Z_constructor();
    };

  Covariance(const str& formula,
             const ArrayXXd &data,
             const strvec& colnames,
             const ArrayXd& parameters) :
    form_(formula), data_(data), colnames_(colnames),
    parameters_(parameters.data(),parameters.data()+parameters.size()),Q_(0), 
    size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()), spchol((make_sparse(),mat)) {
      L_constructor();
      Z_constructor();
    };

  Covariance(const glmmr::Formula& form,
             const ArrayXXd &data,
              const strvec& colnames,
              const ArrayXd& parameters) :
    form_(form), data_(data), colnames_(colnames),
    parameters_(parameters.data(),parameters.data()+parameters.size()),Q_(0), 
    size_B_array((parse(),B_)), dmat_matrix(max_block_dim(),max_block_dim()),
    zquad(max_block_dim()), spchol((make_sparse(),mat)) {
    L_constructor();
    Z_constructor();
  };

  void update_parameters(const dblvec& parameters);
  void update_parameters_extern(const dblvec& parameters);
  void update_parameters(const ArrayXd& parameters);
  void parse();
  double get_val(int b, int i, int j);
  MatrixXd Z();
  MatrixXd D(bool chol = false, bool upper = false);
  VectorXd sim_re();
  double log_likelihood(const VectorXd &u);
  double log_determinant();
  int npar();
  int B();
  int Q();
  int max_block_dim();
  int block_dim(int b);
  void make_sparse();
  MatrixXd ZL();
  MatrixXd LZWZL(const VectorXd& w);
  MatrixXd ZLu(const MatrixXd& u);
  MatrixXd Lu(const MatrixXd& u);
  void set_sparse(bool sparse);
  bool any_group_re();
  intvec parameter_fn_index();
  intvec re_count();
  sparse ZL_sparse();
  sparse Z_sparse();
  strvec parameter_names();
  void derivatives(std::vector<MatrixXd>& derivs,int order = 1);

private:
  std::vector<glmmr::calculator> calc_;
  intvec z_;
  intvec3d re_pars_;
  intvec3d re_cols_data_;
  strvec2d fn_;
  dblvec2d par_for_calcs_;
  std::vector<MatrixXd> re_data_;
  intvec re_fn_par_link_;
  intvec re_count_;
  intvec re_order_;
  int Q_;
  int n_;
  int B_;
  int npars_;
  ArrayXd size_B_array;
  MatrixXd dmat_matrix;
  VectorXd zquad;
  bool isSparse = true;
  sparse mat;
  sparse matZ;
  sparse matL;
  SparseChol spchol;
  int n_threads_;
  
  void update_parameters_in_calculators();
  MatrixXd get_block(int b);
  MatrixXd get_chol_block(int b,bool upper = false);
  MatrixXd D_builder(int b,bool chol = false,bool upper = false);
  void update_ax();
  void L_constructor();
  void Z_constructor();
  MatrixXd D_sparse_builder(bool chol = false,bool upper = false);
  
  
};

}

#include "covariance.ipp"

#endif