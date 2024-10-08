% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{hessian_from_formula}
\alias{hessian_from_formula}
\title{Automatic differentiation of formulae}
\usage{
hessian_from_formula(form_, data_, colnames_, parameters_)
}
\arguments{
\item{form_}{String. Formula to differentiate specified in terms of data items and parameters. Any string not identifying 
a function or a data item names in `colnames` is assumed to be a parameter.}

\item{data_}{Matrix. A matrix including the data. Rows represent observations. The number of columns should match the number 
of items in `colnames_`}

\item{colnames_}{Vector of strings. The names of the columns of `data_`, used to match data named in the formula.}

\item{parameters_}{Vector of doubles. The values of the parameters at which to calculate the derivatives. The parameters should be in the 
same order they appear in the formula.}
}
\value{
A list including the jacobian and hessian matrices.
}
\description{
Exposes the automatic differentiator. Allows for calculation of Jacobian and Hessian matrices 
of formulae in terms of specified parameters. Formula specification is as a string. Data items are automatically 
multiplied by a parameter unless enclosed in parentheses.
}
\examples{
# obtain the Jacobian and Hessian of the log-binomial model log-likelihood. 
# The model is of data from an intervention and control group
# with n1 and n0 participants, respectively, with y1 and y0 the number of events in each group. 
# The mean is exp(alpha) in the control 
# group and exp(alpha + beta) in the intervention group, so that beta is the log relative risk.
hessian_from_formula(
  form_ = "(y1)*(a+b)+((n1)-(y1))*log((1-exp(a+b)))+(y0)*a+((n0)-(y0))*log((1-exp(a)))",
  data_ = matrix(c(10,100,20,100), nrow = 1),
  colnames_ = c("y1","n1","y0","n0"),
  parameters_ = c(log(0.1),log(0.5)))
}
