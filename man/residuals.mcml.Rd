% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/printfunctions.R
\name{residuals.mcml}
\alias{residuals.mcml}
\title{Residuals method for a `mcml` object}
\usage{
\method{residuals}{mcml}(object, type, ...)
}
\arguments{
\item{object}{A `mcml` object.}

\item{type}{Either "standardized", "raw" or "pearson"}

\item{...}{Further arguments passed from other methods}
}
\value{
A matrix with either one column is conditional is false, or with number of columns corresponding 
to the number of MCMC samples.
}
\description{
Calling residuals on an `mcml` object directly is not recommended. This function will currently only generate marginal residuals.
It will generate a new `Model` object internally, thus copying 
all the data, which is not ideal for larger models. The preferred method is to call residuals on either the `Model` 
object or using `Model$residuals()`, both of which will also generate conditional residuals.
}
