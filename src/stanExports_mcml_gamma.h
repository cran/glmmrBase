// Generated by rstantools.  Do not edit by hand.

/*
    glmmr is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    glmmr is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with glmmr.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef MODELS_HPP
#define MODELS_HPP
#define STAN__SERVICES__COMMAND_HPP
#ifndef USE_STANC3
#define USE_STANC3
#endif
#include <rstan/rstaninc.hpp>
// Code generated by stanc v2.32.2
#include <stan/model/model_header.hpp>
namespace model_mcml_gamma_namespace {
using stan::model::model_base_crtp;
using namespace stan::math;
stan::math::profile_map profiles__;
static constexpr std::array<const char*, 21> locations_array__ =
  {" (found before start of program)",
  " (in 'string', line 23, column 2 to column 22)",
  " (in 'string', line 26, column 2 to column 34)",
  " (in 'string', line 27, column 14 to column 85)",
  " (in 'string', line 27, column 2 to column 85)",
  " (in 'string', line 28, column 14 to column 81)",
  " (in 'string', line 28, column 2 to column 81)",
  " (in 'string', line 29, column 14 to column 88)",
  " (in 'string', line 29, column 2 to column 88)",
  " (in 'string', line 14, column 2 to column 8)",
  " (in 'string', line 15, column 2 to column 8)",
  " (in 'string', line 16, column 9 to column 10)",
  " (in 'string', line 16, column 2 to column 15)",
  " (in 'string', line 17, column 9 to column 10)",
  " (in 'string', line 17, column 11 to column 12)",
  " (in 'string', line 17, column 2 to column 16)",
  " (in 'string', line 18, column 8 to column 9)",
  " (in 'string', line 18, column 2 to column 18)",
  " (in 'string', line 19, column 2 to column 15)",
  " (in 'string', line 20, column 2 to column 11)",
  " (in 'string', line 23, column 8 to column 9)"};
#include <stan_meta_header.hpp>
class model_mcml_gamma final : public model_base_crtp<model_mcml_gamma> {
private:
  int N;
  int Q;
  Eigen::Matrix<double,-1,1> Xb_data__;
  Eigen::Matrix<double,-1,-1> Z_data__;
  std::vector<double> y;
  double var_par;
  int type;
  Eigen::Map<Eigen::Matrix<double,-1,1>> Xb{nullptr, 0};
  Eigen::Map<Eigen::Matrix<double,-1,-1>> Z{nullptr, 0, 0};
public:
  ~model_mcml_gamma() {}
  model_mcml_gamma(stan::io::var_context& context__, unsigned int
                   random_seed__ = 0, std::ostream* pstream__ = nullptr)
      : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double;
    boost::ecuyer1988 base_rng__ =
      stan::services::util::create_rng(random_seed__, 0);
    // suppress unused var warning
    (void) base_rng__;
    static constexpr const char* function__ =
      "model_mcml_gamma_namespace::model_mcml_gamma";
    // suppress unused var warning
    (void) function__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      current_statement__ = 9;
      context__.validate_dims("data initialization", "N", "int",
        std::vector<size_t>{});
      N = std::numeric_limits<int>::min();
      current_statement__ = 9;
      N = context__.vals_i("N")[(1 - 1)];
      current_statement__ = 10;
      context__.validate_dims("data initialization", "Q", "int",
        std::vector<size_t>{});
      Q = std::numeric_limits<int>::min();
      current_statement__ = 10;
      Q = context__.vals_i("Q")[(1 - 1)];
      current_statement__ = 11;
      stan::math::validate_non_negative_index("Xb", "N", N);
      current_statement__ = 12;
      context__.validate_dims("data initialization", "Xb", "double",
        std::vector<size_t>{static_cast<size_t>(N)});
      Xb_data__ = Eigen::Matrix<double,-1,1>::Constant(N,
                    std::numeric_limits<double>::quiet_NaN());
      new (&Xb) Eigen::Map<Eigen::Matrix<double,-1,1>>(Xb_data__.data(), N);
      {
        std::vector<local_scalar_t__> Xb_flat__;
        current_statement__ = 12;
        Xb_flat__ = context__.vals_r("Xb");
        current_statement__ = 12;
        pos__ = 1;
        current_statement__ = 12;
        for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
          current_statement__ = 12;
          stan::model::assign(Xb, Xb_flat__[(pos__ - 1)],
            "assigning variable Xb", stan::model::index_uni(sym1__));
          current_statement__ = 12;
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 13;
      stan::math::validate_non_negative_index("Z", "N", N);
      current_statement__ = 14;
      stan::math::validate_non_negative_index("Z", "Q", Q);
      current_statement__ = 15;
      context__.validate_dims("data initialization", "Z", "double",
        std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(Q)});
      Z_data__ = Eigen::Matrix<double,-1,-1>::Constant(N, Q,
                   std::numeric_limits<double>::quiet_NaN());
      new (&Z) Eigen::Map<Eigen::Matrix<double,-1,-1>>(Z_data__.data(), N, Q);
      {
        std::vector<local_scalar_t__> Z_flat__;
        current_statement__ = 15;
        Z_flat__ = context__.vals_r("Z");
        current_statement__ = 15;
        pos__ = 1;
        current_statement__ = 15;
        for (int sym1__ = 1; sym1__ <= Q; ++sym1__) {
          current_statement__ = 15;
          for (int sym2__ = 1; sym2__ <= N; ++sym2__) {
            current_statement__ = 15;
            stan::model::assign(Z, Z_flat__[(pos__ - 1)],
              "assigning variable Z", stan::model::index_uni(sym2__),
              stan::model::index_uni(sym1__));
            current_statement__ = 15;
            pos__ = (pos__ + 1);
          }
        }
      }
      current_statement__ = 16;
      stan::math::validate_non_negative_index("y", "N", N);
      current_statement__ = 17;
      context__.validate_dims("data initialization", "y", "double",
        std::vector<size_t>{static_cast<size_t>(N)});
      y = std::vector<double>(N, std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 17;
      y = context__.vals_r("y");
      current_statement__ = 18;
      context__.validate_dims("data initialization", "var_par", "double",
        std::vector<size_t>{});
      var_par = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 18;
      var_par = context__.vals_r("var_par")[(1 - 1)];
      current_statement__ = 19;
      context__.validate_dims("data initialization", "type", "int",
        std::vector<size_t>{});
      type = std::numeric_limits<int>::min();
      current_statement__ = 19;
      type = context__.vals_i("type")[(1 - 1)];
      current_statement__ = 20;
      stan::math::validate_non_negative_index("gamma", "Q", Q);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = Q;
  }
  inline std::string model_name() const final {
    return "model_mcml_gamma";
  }
  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.32.2",
             "stancflags = --allow-undefined"};
  }
  template <bool propto__, bool jacobian__, typename VecR, typename VecI,
            stan::require_vector_like_t<VecR>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr>
  inline stan::scalar_type_t<VecR>
  log_prob_impl(VecR& params_r__, VecI& params_i__, std::ostream*
                pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    static constexpr const char* function__ =
      "model_mcml_gamma_namespace::log_prob";
    // suppress unused var warning
    (void) function__;
    try {
      std::vector<local_scalar_t__> gamma =
        std::vector<local_scalar_t__>(Q, DUMMY_VAR__);
      current_statement__ = 1;
      gamma = in__.template read<std::vector<local_scalar_t__>>(Q);
      {
        current_statement__ = 2;
        lp_accum__.add(stan::math::std_normal_lpdf<propto__>(
                         stan::math::to_vector(gamma)));
        current_statement__ = 4;
        if (stan::math::logical_eq(type, 1)) {
          current_statement__ = 3;
          lp_accum__.add(stan::math::gamma_lpdf<propto__>(
                           stan::math::to_vector(y), (1 / var_par),
                           stan::math::divide(1,
                             stan::math::multiply(var_par,
                               stan::math::add(Xb,
                                 stan::math::multiply(Z,
                                   stan::math::to_vector(gamma)))))));
        }
        current_statement__ = 6;
        if (stan::math::logical_eq(type, 2)) {
          current_statement__ = 5;
          lp_accum__.add(stan::math::gamma_lpdf<propto__>(
                           stan::math::to_vector(y), (1 / var_par),
                           stan::math::divide(
                             stan::math::add(Xb,
                               stan::math::multiply(Z,
                                 stan::math::to_vector(gamma))), var_par)));
        }
        current_statement__ = 8;
        if (stan::math::logical_eq(type, 3)) {
          current_statement__ = 7;
          lp_accum__.add(stan::math::gamma_lpdf<propto__>(
                           stan::math::to_vector(y), (1 / var_par),
                           stan::math::divide(1,
                             stan::math::multiply(var_par,
                               stan::math::log(
                                 stan::math::add(Xb,
                                   stan::math::multiply(Z,
                                     stan::math::to_vector(gamma))))))));
        }
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
  }
  template <typename RNG, typename VecR, typename VecI, typename VecVar,
            stan::require_vector_like_vt<std::is_floating_point,
            VecR>* = nullptr, stan::require_vector_like_vt<std::is_integral,
            VecI>* = nullptr, stan::require_vector_vt<std::is_floating_point,
            VecVar>* = nullptr>
  inline void
  write_array_impl(RNG& base_rng__, VecR& params_r__, VecI& params_i__,
                   VecVar& vars__, const bool
                   emit_transformed_parameters__ = true, const bool
                   emit_generated_quantities__ = true, std::ostream*
                   pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    // suppress unused var warning
    (void) propto__;
    double lp__ = 0.0;
    // suppress unused var warning
    (void) lp__;
    int current_statement__ = 0;
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    constexpr bool jacobian__ = false;
    static constexpr const char* function__ =
      "model_mcml_gamma_namespace::write_array";
    // suppress unused var warning
    (void) function__;
    try {
      std::vector<double> gamma =
        std::vector<double>(Q, std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 1;
      gamma = in__.template read<std::vector<local_scalar_t__>>(Q);
      out__.write(gamma);
      if (stan::math::logical_negation(
            (stan::math::primitive_value(emit_transformed_parameters__) ||
            stan::math::primitive_value(emit_generated_quantities__)))) {
        return ;
      }
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, typename VecI,
            stan::require_vector_t<VecVar>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr>
  inline void
  unconstrain_array_impl(const VecVar& params_r__, const VecI& params_i__,
                         VecVar& vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      std::vector<local_scalar_t__> gamma =
        std::vector<local_scalar_t__>(Q, DUMMY_VAR__);
      current_statement__ = 1;
      stan::model::assign(gamma, in__.read<std::vector<local_scalar_t__>>(Q),
        "assigning variable gamma");
      out__.write(gamma);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, stan::require_vector_t<VecVar>* = nullptr>
  inline void
  transform_inits_impl(const stan::io::var_context& context__, VecVar&
                       vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      current_statement__ = 1;
      context__.validate_dims("parameter initialization", "gamma", "double",
        std::vector<size_t>{static_cast<size_t>(Q)});
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      std::vector<local_scalar_t__> gamma =
        std::vector<local_scalar_t__>(Q, DUMMY_VAR__);
      current_statement__ = 1;
      gamma = context__.vals_r("gamma");
      out__.write(gamma);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  inline void
  get_param_names(std::vector<std::string>& names__, const bool
                  emit_transformed_parameters__ = true, const bool
                  emit_generated_quantities__ = true) const {
    names__ = std::vector<std::string>{"gamma"};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline void
  get_dims(std::vector<std::vector<size_t>>& dimss__, const bool
           emit_transformed_parameters__ = true, const bool
           emit_generated_quantities__ = true) const {
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{static_cast<
                                                                    size_t>(Q)}};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline void
  constrained_param_names(std::vector<std::string>& param_names__, bool
                          emit_transformed_parameters__ = true, bool
                          emit_generated_quantities__ = true) const final {
    for (int sym1__ = 1; sym1__ <= Q; ++sym1__) {
      param_names__.emplace_back(std::string() + "gamma" + '.' +
        std::to_string(sym1__));
    }
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline void
  unconstrained_param_names(std::vector<std::string>& param_names__, bool
                            emit_transformed_parameters__ = true, bool
                            emit_generated_quantities__ = true) const final {
    for (int sym1__ = 1; sym1__ <= Q; ++sym1__) {
      param_names__.emplace_back(std::string() + "gamma" + '.' +
        std::to_string(sym1__));
    }
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {}
  }
  inline std::string get_constrained_sizedtypes() const {
    return std::string("[{\"name\":\"gamma\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(Q) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"}]");
  }
  inline std::string get_unconstrained_sizedtypes() const {
    return std::string("[{\"name\":\"gamma\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(Q) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"}]");
  }
  // Begin method overload boilerplate
  template <typename RNG> inline void
  write_array(RNG& base_rng, Eigen::Matrix<double,-1,1>& params_r,
              Eigen::Matrix<double,-1,1>& vars, const bool
              emit_transformed_parameters = true, const bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = Q;
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * (0);
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    std::vector<int> params_i;
    vars = Eigen::Matrix<double,-1,1>::Constant(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <typename RNG> inline void
  write_array(RNG& base_rng, std::vector<double>& params_r, std::vector<int>&
              params_i, std::vector<double>& vars, bool
              emit_transformed_parameters = true, bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = Q;
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * (0);
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    vars = std::vector<double>(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(Eigen::Matrix<T_,-1,1>& params_r, std::ostream* pstream = nullptr) const {
    Eigen::Matrix<int,-1,1> params_i;
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(std::vector<T_>& params_r, std::vector<int>& params_i,
           std::ostream* pstream = nullptr) const {
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  inline void
  transform_inits(const stan::io::var_context& context,
                  Eigen::Matrix<double,-1,1>& params_r, std::ostream*
                  pstream = nullptr) const final {
    std::vector<double> params_r_vec(params_r.size());
    std::vector<int> params_i;
    transform_inits(context, params_i, params_r_vec, pstream);
    params_r = Eigen::Map<Eigen::Matrix<double,-1,1>>(params_r_vec.data(),
                 params_r_vec.size());
  }
  inline void
  transform_inits(const stan::io::var_context& context, std::vector<int>&
                  params_i, std::vector<double>& vars, std::ostream*
                  pstream__ = nullptr) const {
    vars.resize(num_params_r__);
    transform_inits_impl(context, vars, pstream__);
  }
  inline void
  unconstrain_array(const std::vector<double>& params_constrained,
                    std::vector<double>& params_unconstrained, std::ostream*
                    pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = std::vector<double>(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
  inline void
  unconstrain_array(const Eigen::Matrix<double,-1,1>& params_constrained,
                    Eigen::Matrix<double,-1,1>& params_unconstrained,
                    std::ostream* pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = Eigen::Matrix<double,-1,1>::Constant(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
};
}
using stan_model = model_mcml_gamma_namespace::model_mcml_gamma;
#ifndef USING_R
// Boilerplate
stan::model::model_base&
new_model(stan::io::var_context& data_context, unsigned int seed,
          std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}
stan::math::profile_map& get_stan_profile_data() {
  return model_mcml_gamma_namespace::profiles__;
}
#endif
#endif
