% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/lme4_functions.R
\name{mcml_lmer}
\alias{mcml_lmer}
\title{lme4 style linear mixed model}
\usage{
mcml_lmer(
  formula,
  data,
  start = NULL,
  offset = NULL,
  verbose = 1L,
  iter.warmup = 100,
  iter.sampling = 50,
  weights = NULL,
  ...
)
}
\arguments{
\item{formula}{A two-sided linear formula object including both the fixed and random effects specifications, see Details.}

\item{data}{A data frame containing the variables named in `formula`.}

\item{start}{Optional. A vector of starting values for the fixed effects.}

\item{offset}{Optional. A vector of offset values.}

\item{verbose}{Integer, controls the level of detail printed to the console, either 0 (no output), 
1 (main output), or 2 (detailed output)}

\item{iter.warmup}{The number of warmup iterations for the MCMC sampling step of each iteration.}

\item{iter.sampling}{The number of sampling iterations for the MCMC sampling step of each iteration.}

\item{weights}{Optional. A vector of observation level weights to apply to the model fit.}

\item{...}{additional arguments passed to `Model$MCML()`}
}
\value{
A `mcml` model fit object.
}
\description{
A wrapper for Model stochastic maximum likelihood model fitting replicating lme4's syntax
}
\details{
This function aims to replicate the syntax of lme4's `lmer` command. The specified formula can be 
the standard lme4 syntax, or alternatively a glmmrBase style formula can also be used to allow for the 
wider range of covariance function specifications. For example both `y~x+(1|cl/t)` and `y~x+(1|gr(cl))+(1|gr(cl)*ar1(t))`
would be valid formulae.
}
\examples{
#create a data frame describing a cross-sectional parallel cluster
#randomised trial
df <- nelder(~(cl(10)*t(5)) > ind(10))
df$int <- 0
df[df$cl > 5, 'int'] <- 1
# simulate data using the Model class
df$y <- Model$new(
  formula = ~ factor(t) + int - 1 + (1|gr(cl)) + (1|gr(cl,t)),
  data = df,
  family = stats::gaussian()
)$sim_data()
\dontrun{
fit <- mcml_lmer(y ~ factor(t) + int - 1 + (1|cl/t), data = df)
}
}
