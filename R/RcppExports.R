# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

Covariance__new <- function(form_, data_, colnames_) {
    .Call(`_glmmrBase_Covariance__new`, form_, data_, colnames_)
}

Covariance_nngp__new <- function(form_, data_, colnames_) {
    .Call(`_glmmrBase_Covariance_nngp__new`, form_, data_, colnames_)
}

Covariance_hsgp__new <- function(form_, data_, colnames_) {
    .Call(`_glmmrBase_Covariance_hsgp__new`, form_, data_, colnames_)
}

Covariance__Z <- function(xp, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__Z`, xp, type_)
}

Covariance__ZL <- function(xp, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__ZL`, xp, type_)
}

Covariance__LZWZL <- function(xp, w_, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__LZWZL`, xp, w_, type_)
}

Covariance__Update_parameters <- function(xp, parameters_, type_ = 0L) {
    invisible(.Call(`_glmmrBase_Covariance__Update_parameters`, xp, parameters_, type_))
}

Covariance__D <- function(xp, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__D`, xp, type_)
}

Covariance__D_chol <- function(xp, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__D_chol`, xp, type_)
}

Covariance__B <- function(xp, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__B`, xp, type_)
}

Covariance__Q <- function(xp, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__Q`, xp, type_)
}

Covariance__log_likelihood <- function(xp, u_, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__log_likelihood`, xp, u_, type_)
}

Covariance__log_determinant <- function(xp, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__log_determinant`, xp, type_)
}

Covariance__n_cov_pars <- function(xp, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__n_cov_pars`, xp, type_)
}

Covariance__simulate_re <- function(xp, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__simulate_re`, xp, type_)
}

Covariance__make_sparse <- function(xp, amd = TRUE, type_ = 0L) {
    invisible(.Call(`_glmmrBase_Covariance__make_sparse`, xp, amd, type_))
}

Covariance__make_dense <- function(xp, type_ = 0L) {
    invisible(.Call(`_glmmrBase_Covariance__make_dense`, xp, type_))
}

Covariance__set_nn <- function(xp, nn) {
    invisible(.Call(`_glmmrBase_Covariance__set_nn`, xp, nn))
}

Covariance__any_gr <- function(xp, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__any_gr`, xp, type_)
}

Covariance__get_val <- function(xp, i, j, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__get_val`, xp, i, j, type_)
}

Covariance__parameter_fn_index <- function(xp, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__parameter_fn_index`, xp, type_)
}

Covariance__re_terms <- function(xp, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__re_terms`, xp, type_)
}

Covariance__re_count <- function(xp, type_ = 0L) {
    .Call(`_glmmrBase_Covariance__re_count`, xp, type_)
}

Linpred__new <- function(formula_, data_, colnames_) {
    .Call(`_glmmrBase_Linpred__new`, formula_, data_, colnames_)
}

Linpred__update_pars <- function(xp, parameters_) {
    invisible(.Call(`_glmmrBase_Linpred__update_pars`, xp, parameters_))
}

Linpred__xb <- function(xp) {
    .Call(`_glmmrBase_Linpred__xb`, xp)
}

Linpred__x <- function(xp) {
    .Call(`_glmmrBase_Linpred__x`, xp)
}

Linpred__beta_names <- function(xp) {
    .Call(`_glmmrBase_Linpred__beta_names`, xp)
}

Linpred__any_nonlinear <- function(xp) {
    .Call(`_glmmrBase_Linpred__any_nonlinear`, xp)
}

ModelBits__new <- function(formula_, data_, colnames_, family_, link_, beta_, theta_) {
    .Call(`_glmmrBase_ModelBits__new`, formula_, data_, colnames_, family_, link_, beta_, theta_)
}

ModelBits__update_beta <- function(xp, beta_) {
    invisible(.Call(`_glmmrBase_ModelBits__update_beta`, xp, beta_))
}

ModelBits__update_theta <- function(xp, theta_) {
    invisible(.Call(`_glmmrBase_ModelBits__update_theta`, xp, theta_))
}

Model__new_w_pars <- function(formula_, data_, colnames_, family_, link_, beta_, theta_) {
    .Call(`_glmmrBase_Model__new_w_pars`, formula_, data_, colnames_, family_, link_, beta_, theta_)
}

Model__new <- function(formula_, data_, colnames_, family_, link_) {
    .Call(`_glmmrBase_Model__new`, formula_, data_, colnames_, family_, link_)
}

Model_nngp__new <- function(formula_, data_, colnames_, family_, link_) {
    .Call(`_glmmrBase_Model_nngp__new`, formula_, data_, colnames_, family_, link_)
}

Model_nngp__new_w_pars <- function(formula_, data_, colnames_, family_, link_, beta_, theta_, nn) {
    .Call(`_glmmrBase_Model_nngp__new_w_pars`, formula_, data_, colnames_, family_, link_, beta_, theta_, nn)
}

Model_hsgp__new <- function(formula_, data_, colnames_, family_, link_) {
    .Call(`_glmmrBase_Model_hsgp__new`, formula_, data_, colnames_, family_, link_)
}

Model_hsgp__new_w_pars <- function(formula_, data_, colnames_, family_, link_, beta_, theta_) {
    .Call(`_glmmrBase_Model_hsgp__new_w_pars`, formula_, data_, colnames_, family_, link_, beta_, theta_)
}

Model__set_y <- function(xp, y_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__set_y`, xp, y_, type))
}

Model__set_offset <- function(xp, offset_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__set_offset`, xp, offset_, type))
}

Model__set_weights <- function(xp, weights_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__set_weights`, xp, weights_, type))
}

Model__P <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__P`, xp, type)
}

Model__Q <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__Q`, xp, type)
}

Model__theta_size <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__theta_size`, xp, type)
}

Model__update_beta <- function(xp, beta_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__update_beta`, xp, beta_, type))
}

Model__update_theta <- function(xp, theta_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__update_theta`, xp, theta_, type))
}

Model__update_u <- function(xp, u_, append = FALSE, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__update_u`, xp, u_, append, type))
}

Model__set_quantile <- function(xp, q, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__set_quantile`, xp, q, type))
}

Model__use_attenuation <- function(xp, use_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__use_attenuation`, xp, use_, type))
}

Model__update_W <- function(xp, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__update_W`, xp, type))
}

Model__get_W <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__get_W`, xp, type)
}

Model__set_direct_control <- function(xp, direct = FALSE, direct_range_beta = 3.0, max_iter = 100L, epsilon = 1e-4, select_one = TRUE, trisect_once = FALSE, max_eval = 0L, mrdirect = FALSE, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__set_direct_control`, xp, direct, direct_range_beta, max_iter, epsilon, select_one, trisect_once, max_eval, mrdirect, type))
}

Model__set_lbfgs_control <- function(xp, g_epsilon = 1e-8, past = 3L, delta = 1e-8, max_linesearch = 64L, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__set_lbfgs_control`, xp, g_epsilon, past, delta, max_linesearch, type))
}

Model__set_bound <- function(xp, bound_, beta = TRUE, lower = TRUE, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__set_bound`, xp, bound_, beta, lower, type))
}

Model__print_instructions <- function(xp, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__print_instructions`, xp, type))
}

Model__log_prob <- function(xp, v_, type = 0L) {
    .Call(`_glmmrBase_Model__log_prob`, xp, v_, type)
}

Model__set_bobyqa_control <- function(xp, npt_, rhobeg_, rhoend_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__set_bobyqa_control`, xp, npt_, rhobeg_, rhoend_, type))
}

Model__log_gradient <- function(xp, v_, beta_, type = 0L) {
    .Call(`_glmmrBase_Model__log_gradient`, xp, v_, beta_, type)
}

Model__linear_predictor <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__linear_predictor`, xp, type)
}

Model__log_likelihood <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__log_likelihood`, xp, type)
}

Model__cov_set_nn <- function(xp, nn) {
    invisible(.Call(`_glmmrBase_Model__cov_set_nn`, xp, nn))
}

Model__test_lbfgs <- function(xp, x) {
    invisible(.Call(`_glmmrBase_Model__test_lbfgs`, xp, x))
}

Model__test_lbfgs_theta <- function(xp, x) {
    invisible(.Call(`_glmmrBase_Model__test_lbfgs_theta`, xp, x))
}

Model__test_lbfgs_laplace <- function(xp, x) {
    invisible(.Call(`_glmmrBase_Model__test_lbfgs_laplace`, xp, x))
}

Model__ml_beta <- function(xp, algo = 0L, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__ml_beta`, xp, algo, type))
}

Model__ml_theta <- function(xp, algo = 0L, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__ml_theta`, xp, algo, type))
}

Model__ml_all <- function(xp, algo = 0L, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__ml_all`, xp, algo, type))
}

Model__laplace_ml_beta_u <- function(xp, algo = 0L, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__laplace_ml_beta_u`, xp, algo, type))
}

Model__laplace_ml_theta <- function(xp, algo = 0L, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__laplace_ml_theta`, xp, algo, type))
}

Model__laplace_ml_beta_theta <- function(xp, algo = 0L, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__laplace_ml_beta_theta`, xp, algo, type))
}

Model__nr_beta <- function(xp, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__nr_beta`, xp, type))
}

Model__laplace_nr_beta_u <- function(xp, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__laplace_nr_beta_u`, xp, type))
}

Model__Sigma <- function(xp, inverse, type = 0L) {
    .Call(`_glmmrBase_Model__Sigma`, xp, inverse, type)
}

Model__information_matrix <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__information_matrix`, xp, type)
}

Model__D <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__D`, xp, type)
}

Model__D_chol <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__D_chol`, xp, type)
}

Model__u_log_likelihood <- function(xp, u_, type = 0L) {
    .Call(`_glmmrBase_Model__u_log_likelihood`, xp, u_, type)
}

Model__simulate_re <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__simulate_re`, xp, type)
}

Model__re_terms <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__re_terms`, xp, type)
}

Model__re_count <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__re_count`, xp, type)
}

Model__parameter_fn_index <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__parameter_fn_index`, xp, type)
}

Model__information_matrix_crude <- function(xp, type = 2L) {
    .Call(`_glmmrBase_Model__information_matrix_crude`, xp, type)
}

Model__obs_information_matrix <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__obs_information_matrix`, xp, type)
}

Model__u <- function(xp, scaled_, type = 0L) {
    .Call(`_glmmrBase_Model__u`, xp, scaled_, type)
}

Model__Zu <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__Zu`, xp, type)
}

Model__X <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__X`, xp, type)
}

Model__mcmc_sample <- function(xp, warmup_, samples_, adapt_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__mcmc_sample`, xp, warmup_, samples_, adapt_, type))
}

Model__set_trace <- function(xp, trace_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__set_trace`, xp, trace_, type))
}

Model__get_beta <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__get_beta`, xp, type)
}

Model__y <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__y`, xp, type)
}

Model__get_theta <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__get_theta`, xp, type)
}

Model__get_var_par <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__get_var_par`, xp, type)
}

Model__get_variance <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__get_variance`, xp, type)
}

Model__set_var_par <- function(xp, var_par_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__set_var_par`, xp, var_par_, type))
}

Model__set_trials <- function(xp, trials, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__set_trials`, xp, trials, type))
}

Model__L <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__L`, xp, type)
}

Model__ZL <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__ZL`, xp, type)
}

Model__xb <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__xb`, xp, type)
}

near_semi_pd <- function(mat_) {
    .Call(`_glmmrBase_near_semi_pd`, mat_)
}

Covariance__submatrix <- function(xp, i) {
    .Call(`_glmmrBase_Covariance__submatrix`, xp, i)
}

Model_hsgp__set_approx_pars <- function(xp, m_, L_) {
    invisible(.Call(`_glmmrBase_Model_hsgp__set_approx_pars`, xp, m_, L_))
}

Covariance_hsgp__set_approx_pars <- function(xp, m_, L_) {
    invisible(.Call(`_glmmrBase_Covariance_hsgp__set_approx_pars`, xp, m_, L_))
}

Model_hsgp__dim <- function(xp) {
    .Call(`_glmmrBase_Model_hsgp__dim`, xp)
}

Model__aic <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__aic`, xp, type)
}

Model__residuals <- function(xp, rtype = 2L, conditional = TRUE, type = 0L) {
    .Call(`_glmmrBase_Model__residuals`, xp, rtype, conditional, type)
}

Model__get_log_likelihood_values <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__get_log_likelihood_values`, xp, type)
}

Model__u_diagnostic <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__u_diagnostic`, xp, type)
}

Model__marginal <- function(xp, x, margin = 0L, re = 3L, se = 0L, at = NULL, atmeans = NULL, average = NULL, xvals_first = 1, xvals_second = 0, atvals = NULL, revals = NULL, type = 0L) {
    .Call(`_glmmrBase_Model__marginal`, xp, x, margin, re, se, at, atmeans, average, xvals_first, xvals_second, atvals, revals, type)
}

Model__mcmc_set_lambda <- function(xp, lambda_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__mcmc_set_lambda`, xp, lambda_, type))
}

Model__reset_fn_counter <- function(xp, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__reset_fn_counter`, xp, type))
}

Model__get_fn_counter <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__get_fn_counter`, xp, type)
}

Model__print_names <- function(xp, data, parameters, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__print_names`, xp, data, parameters, type))
}

Model__mcmc_set_max_steps <- function(xp, max_steps_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__mcmc_set_max_steps`, xp, max_steps_, type))
}

Model__set_sml_parameters <- function(xp, saem_, block_size = 20L, alpha = 0.8, pr_average = TRUE, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__set_sml_parameters`, xp, saem_, block_size, alpha, pr_average, type))
}

Model__ll_diff_variance <- function(xp, beta, theta, type = 0L) {
    .Call(`_glmmrBase_Model__ll_diff_variance`, xp, beta, theta, type)
}

Model__mcmc_set_refresh <- function(xp, refresh_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__mcmc_set_refresh`, xp, refresh_, type))
}

Model__mcmc_set_target_accept <- function(xp, target_, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__mcmc_set_target_accept`, xp, target_, type))
}

Model__make_sparse <- function(xp, amd = TRUE, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__make_sparse`, xp, amd, type))
}

Model__make_dense <- function(xp, type = 0L) {
    invisible(.Call(`_glmmrBase_Model__make_dense`, xp, type))
}

Model__beta_parameter_names <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__beta_parameter_names`, xp, type)
}

Model__theta_parameter_names <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__theta_parameter_names`, xp, type)
}

Model__hessian_correction <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__hessian_correction`, xp, type)
}

Model__any_nonlinear <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__any_nonlinear`, xp, type)
}

Model__sandwich <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__sandwich`, xp, type)
}

Model__infomat_theta <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__infomat_theta`, xp, type)
}

Model__kenward_roger <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__kenward_roger`, xp, type)
}

Model__small_sample_correction <- function(xp, ss_type = 0L, type = 0L) {
    .Call(`_glmmrBase_Model__small_sample_correction`, xp, ss_type, type)
}

Model__box <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__box`, xp, type)
}

Model__cov_deriv <- function(xp, type = 0L) {
    .Call(`_glmmrBase_Model__cov_deriv`, xp, type)
}

Model__predict <- function(xp, newdata_, newoffset_, m, type = 0L) {
    .Call(`_glmmrBase_Model__predict`, xp, newdata_, newoffset_, m, type)
}

Model__predict_re <- function(xp, newdata_, newoffset_, m, type = 0L) {
    .Call(`_glmmrBase_Model__predict_re`, xp, newdata_, newoffset_, m, type)
}

#' Automatic differentiation of formulae
#' 
#' Exposes the automatic differentiator. Allows for calculation of Jacobian and Hessian matrices 
#' of formulae in terms of specified parameters. Formula specification is as a string. Data items are automatically 
#' multiplied by a parameter unless enclosed in parentheses.
#' @param form_ String. Formula to differentiate specified in terms of data items and parameters. Any string not identifying 
#' a function or a data item names in `colnames` is assumed to be a parameter.
#' @param data_ Matrix. A matrix including the data. Rows represent observations. The number of columns should match the number 
#' of items in `colnames_`
#' @param colnames_ Vector of strings. The names of the columns of `data_`, used to match data named in the formula.
#' @param parameters_ Vector of doubles. The values of the parameters at which to calculate the derivatives. The parameters should be in the 
#' same order they appear in the formula.
#' @return A list including the jacobian and hessian matrices.
#' @examples
#' # obtain the Jacobian and Hessian of the log-binomial model log-likelihood. 
#' # The model is of data from an intervention and control group
#' # with n1 and n0 participants, respectively, with y1 and y0 the number of events in each group. 
#' # The mean is exp(alpha) in the control 
#' # group and exp(alpha + beta) in the intervention group, so that beta is the log relative risk.
#' hessian_from_formula(
#'   form_ = "(y1)*(a+b)+((n1)-(y1))*log((1-exp(a+b)))+(y0)*a+((n0)-(y0))*log((1-exp(a)))",
#'   data_ = matrix(c(10,100,20,100), nrow = 1),
#'   colnames_ = c("y1","n1","y0","n0"),
#'   parameters_ = c(log(0.1),log(0.5)))
#' @export
hessian_from_formula <- function(form_, data_, colnames_, parameters_) {
    .Call(`_glmmrBase_hessian_from_formula`, form_, data_, colnames_, parameters_)
}

#' Disable or enable parallelised computing
#' 
#' By default, the package will use multithreading for many calculations if OpenMP is 
#' available on the system. For multi-user systems this may not be desired, so parallel
#' execution can be disabled with this function.
#' 
#' @param parallel_ Logical indicating whether to use parallel computation (TRUE) or disable it (FALSE)
#' @param cores_ Number of cores for parallel execution
#' @return None, called for effects
setParallel <- function(parallel_, cores_ = 2L) {
    invisible(.Call(`_glmmrBase_setParallel`, parallel_, cores_))
}

re_names <- function(formula, as_formula = TRUE) {
    .Call(`_glmmrBase_re_names`, formula, as_formula)
}

attenuate_xb <- function(xb, Z, D, link) {
    .Call(`_glmmrBase_attenuate_xb`, xb, Z, D, link)
}

dlinkdeta <- function(xb, link) {
    .Call(`_glmmrBase_dlinkdeta`, xb, link)
}

girling_algorithm <- function(xp, N_, C_, tol_) {
    .Call(`_glmmrBase_girling_algorithm`, xp, N_, C_, tol_)
}

get_variable_names <- function(formula_, colnames_) {
    .Call(`_glmmrBase_get_variable_names`, formula_, colnames_)
}

