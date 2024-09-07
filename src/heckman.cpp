// #include <RcppEigen.h>
// #include <boost/math/distributions/normal.hpp>
// #include "glmmr.h"
// 
// using namespace Rcpp;
// using namespace Eigen;
// 
// double heckman_ll(const double mu1,
//                   const double mu2,
//                   const double n,
//                   const double ycont,
//                   const double ycount)
// {
//   boost::math::normal norm(0, 1);
//   double lfk = glmmr::maths::log_factorial_approx(ycount);
//   double lfn = glmmr::maths::log_factorial_approx(n);
//   double lfnk = glmmr::maths::log_factorial_approx(n - ycount);
//   double logl_probit = lfn - lfk - lfnk + ycount*((double)cdf(norm,mu2)) + (n - ycount)*log(1 - (double)cdf(norm,mu2));
//   double logl_cont = -0.5*log(var_par) -0.5*log(2*3.141593) - 0.5*(y - mu)*(y - mu)/var_par;
//   
//   return logl_cont + logl_probit;
// }
// 
// class heckman {
//   public:
//     VectorXd ycont;
//     VectorXd ycount;
//     MatrixXd X1;
//     MatrixXd X2;
//     MatrixXd alpha;
//     MatrixXd v;
//     VectorXd xb1;
//     VectorXd xb2;
//     double theta = 0;
//     
//     heckman(const VectorXd& ycont_,
//             const VectorXd& ycount_,
//             const MatrixXd& X1_,
//             const MatrixXd& X2_,
//             const MatrixXd& alpha_,
//             const MatrixXd v_) : ycont(ycont_), ycount(ycount_), X1(X1_), X2(X2_), alpha(alpha_), v(v_), mu1(X1_.rows()), mu2(X2_.rows()) {};
//     
//     void update_beta(const dblvec& beta){
//       int n1 = X1.cols();
//       int n2 = X2.cols();
//       VectorXd beta1(beta.data(), n1);
//       VectorXd beta2(beta.data()+n1, n2);
//       mu1 = X1*beta1;
//       mu2 = X2*beta2;
//       theta = beta[n1+n2];
//     }
//     
//     double ll_beta(const dblvec& beta){
//       update_beta(beta);
//       int n = X1.rows();
//       int iter = alpha.cols();
//       double ll = 0;
//       for(int i = 0; i < n; i++){
//         for(int j = 0; j < iter; j++){
//           ll += heckman_ll(mu1(i) + theta*alpha(i,j) + v(i,j),
//                            mu2(i) + alpha(i,j))
//         }
//       }
//     }
//     
// };
// 
// 
// 
// // [[Rcpp::export]]
// double heckman_beta(
//                     )
// {
//   
// }
