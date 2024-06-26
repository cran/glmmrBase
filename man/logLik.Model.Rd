% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/model_s3_functions.R
\name{logLik.Model}
\alias{logLik.Model}
\title{Extracts the log-likelihood from an mcml object}
\usage{
\method{logLik}{Model}(object, ...)
}
\arguments{
\item{object}{An `Model` object.}

\item{...}{Further arguments passed from other methods}
}
\value{
An object of class `logLik`. If both `fixed` and `covariance` are FALSE then it returns NA.
}
\description{
Extracts the log-likelihood value from an `Model` object. If no data `y` are specified then it returns NA.
}
