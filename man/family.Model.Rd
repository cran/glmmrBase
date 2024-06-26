% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/model_s3_functions.R
\name{family.Model}
\alias{family.Model}
\title{Extracts the family from a `Model` object. This information can also be
accessed directly from the Model as `Model$family`}
\usage{
\method{family}{Model}(object, ...)
}
\arguments{
\item{object}{A `Model` object.}

\item{...}{Further arguments passed from other methods}
}
\value{
A \link[stats]{family} object.
}
\description{
Extracts the \link[stats]{family} from a `Model` object.
}
