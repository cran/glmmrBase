% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/steppedwedge.R
\name{stepped_wedge}
\alias{stepped_wedge}
\title{Generate a stepped-wedge design}
\usage{
stepped_wedge(
  J,
  M,
  nper = 1,
  beta = c(rep(0, J + 1), 0),
  icc,
  cac = NULL,
  iac = NULL,
  var = 1,
  family = stats::gaussian()
)
}
\arguments{
\item{J}{Integer indicating the number of sequences such that there are J+1 time periods}

\item{M}{Integer. The number of individual observations per cluster-period, assumed equal across all clusters}

\item{nper}{Integer. The number of clusters per sequence, default is one.}

\item{beta}{Vector of beta parameters to initialise the design, defaults to all zeros.}

\item{icc}{Intraclass correlation coefficient. User may specify
more than one value, see details.}

\item{cac}{Cluster autocorrelation coefficient, optional and user may specify more than one value, see details}

\item{iac}{Individual autocorrelation coefficient, optional and user may specify more than one value, see details}

\item{var}{Assumed overall variance of the model, used to calculate the other covariance, see details}

\item{family}{a \link[stats]{family} object}
}
\value{
A Model object with MeanFunction and Covariance objects, or
a ModelSpace holding several such Model objects.
}
\description{
Generate a stepped-wedge cluster randomised trial design in glmmr
}
\details{
The complete stepped-wedge cluster randomised trial design has J sequences of clusters
observed over J+1 time periods, each sequence has `nper` clusters. 
The first time period has all clusters in the control state, and the final time period has
all clusters in the treatment state, with one sequence switching between the two each
period to create the "step".


The assumed generalised linear mixed model for the stepped-wedge cluster trial is, for 
individual i, in cluster j, at time t:

\deqn{y_{ijt} \sim F(\mu_{ijt},\sigma)}
\deqn{\mu_{ijt} = h^-1(x_{ijt}\beta + \alpha_{1j} + \alpha_{2jt} + \alpha_{3i})}
\deqn{\alpha_{p.} \sim N(0,\sigma^2_p), p = 1,2,3}

Defining \eqn{\tau} as the total model variance, then the intraclass correlation 
coefficient (ICC) is
\deqn{ICC = \frac{\sigma_1 + \sigma_2}{\tau}}
the cluster autocorrelation coefficient (CAC) is :
\deqn{CAC = \frac{\sigma_1}{\sigma_1 + \sigma_2}}
and the individual autocorrelation coefficient as:
\deqn{IAC = \frac{\sigma_3}{\tau(1-ICC)}}

When CAC and/or IAC are not specified in the call, then the respective random effects
terms are assumed to be zero. For example, if IAC is not specified then \eqn{\alpha_{3i}}
does not appear in the model, and we have a cross-sectional sampling design; if IAC
were specified then we would have a cohort.

For non-linear models, such as Poisson or Binomial models, there is no single obvious choice
for `var_par` (\eqn{\tau} in the above formulae), as the models are heteroskedastic. Choices 
might include the variance at the mean values of the parameters or a reasonable choice based
on the variance of the respective distribution.

If the user specifies more than one value for icc, cac, or iac, then a ModelSpace is returned
with Models with every combination of parameters. This can be used in particular to generate
a design space for optimal design analyses.
}
\examples{
#generate a simple design with only cluster random effects and 6 clusters with 10
#individuals in each cluster-period
des <- stepped_wedge(6,10,icc=0.05)
# same design but with a cohort of individuals
des <- stepped_wedge(6,10,icc=0.05, iac = 0.1)
# same design, but with two clusters per sequence and specifying the initial parameters
des <- stepped_wedge(6,10,beta = c(rnorm(7,0,0.1),-0.1),icc=0.05, iac = 0.1)
# specifying multiple values of the variance parameters will return a design space 
# with all designs with all the combinations of the variance parameter
des <- stepped_wedge(6,10,icc=c(0.01,0.05), cac = c(0.5,0.7,0.9), iac = 0.1)
}
\seealso{
\link[glmmrBase]{Model}
}
