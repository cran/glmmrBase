% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/R6meanfunction.R
\name{MeanFunction}
\alias{MeanFunction}
\title{R6 Class representing a mean function/linear predictor}
\description{
R6 Class representing a mean function/linear predictor

R6 Class representing a mean function/linear predictor
}
\details{
For the generalised linear mixed model 

\deqn{Y \sim F(\mu,\sigma)}
\deqn{\mu = h^-1(X\beta + Z\gamma)}
\deqn{\gamma \sim MVN(0,D)}

this class defines the fixed effects design matrix X. 
The mean function is defined by a model formula, data, and parameters.
A new instance can be generated with $new(). The class will generate the 
relevant matrix X automatically. See \href{https://github.com/samuel-watson/glmmrBase/blob/master/README.md}{glmmrBase} for a 
detailed guide on model specification.


Specification of the mean function follows standard model formulae in R. 
For example for a stepped-wedge cluster trial model, a typical mean model is 
\eqn{E(y_{ijt}|\delta)=\beta_0 + \tau_t + \beta_1 d_{jt} + z_{ijt}\delta} where \eqn{\tau_t} 
are fixed effects for each time period. The formula specification for this would be `~ factor(t) + int` 
where `int` is the name of the variable indicating the treatment.

One can also include non-linear functions of variables in the mean function, and name the parameters. 
The resulting X matrix is then a matrix of first-order partial derivatives. For example, one can
specify `~ int + b_1*exp(b_2*x)`.



Using `update_parameters()` is the preferred way of updating the parameters of the 
mean or covariance objects as opposed to direct assignment, e.g. `self$parameters <- c(...)`. 
The function calls check functions to automatically update linked matrices with the new parameters.
}
\examples{

## ------------------------------------------------
## Method `MeanFunction$n`
## ------------------------------------------------

\dontshow{
setParallel(FALSE) # for the CRAN check
}
df <- nelder(~(cl(4)*t(5)) > ind(5))
df$int <- 0
df[df$cl <= 2, 'int'] <- 1
mf1 <- MeanFunction$new(formula = ~ int ,
                        data=df,
                        parameters = c(-1,1)
                        )
mf1$n()

## ------------------------------------------------
## Method `MeanFunction$new`
## ------------------------------------------------

\dontshow{
setParallel(FALSE) # for the CRAN check
}
df <- nelder(~(cl(4)*t(5)) > ind(5))
df$int <- 0
df[df$cl <= 2, 'int'] <- 1
mf1 <- MeanFunction$new(formula = ~ int ,
                        data=df,
                        parameters = c(-1,1),
                        )

## ------------------------------------------------
## Method `MeanFunction$colnames`
## ------------------------------------------------

\dontshow{
setParallel(FALSE) # for the CRAN check
}
df <- nelder(~(cl(4)*t(5)) > ind(5))
df$int <- 0
df[df$cl <= 5, 'int'] <- 1
mf1 <- MeanFunction$new(formula = ~ int ,
                        data=df,
                        parameters = c(-1,1)
                        )
mf1$colnames(c("cluster","time","individual","treatment"))
mf1$colnames()

## ------------------------------------------------
## Method `MeanFunction$subset_rows`
## ------------------------------------------------

\dontshow{
setParallel(FALSE) # for the CRAN check
}
df <- nelder(~(cl(4)*t(5)) > ind(5))
df$int <- 0
df[df$cl <= 5, 'int'] <- 1
mf1 <- MeanFunction$new(formula = ~ int ,
                        data=df,
                        parameters = c(-1,1)
                        )
mf1$subset_rows(1:20) 
}
\section{Public fields}{
\if{html}{\out{<div class="r6-fields">}}
\describe{
\item{\code{formula}}{model formula for the fixed effects}

\item{\code{data}}{Data frame with data required to build X}

\item{\code{parameters}}{A vector of parameter values for \eqn{\beta} used for simulating data and calculating
covariance matrix of observations for non-linear models.}

\item{\code{offset}}{An optional vector specifying the offset values}

\item{\code{X}}{the fixed effects design matrix}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-MeanFunction-n}{\code{MeanFunction$n()}}
\item \href{#method-MeanFunction-new}{\code{MeanFunction$new()}}
\item \href{#method-MeanFunction-print}{\code{MeanFunction$print()}}
\item \href{#method-MeanFunction-update_parameters}{\code{MeanFunction$update_parameters()}}
\item \href{#method-MeanFunction-colnames}{\code{MeanFunction$colnames()}}
\item \href{#method-MeanFunction-subset_rows}{\code{MeanFunction$subset_rows()}}
\item \href{#method-MeanFunction-linear_predictor}{\code{MeanFunction$linear_predictor()}}
\item \href{#method-MeanFunction-any_nonlinear}{\code{MeanFunction$any_nonlinear()}}
\item \href{#method-MeanFunction-clone}{\code{MeanFunction$clone()}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-MeanFunction-n"></a>}}
\if{latex}{\out{\hypertarget{method-MeanFunction-n}{}}}
\subsection{Method \code{n()}}{
Returns the number of observations
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{MeanFunction$n()}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{...}}{ignored}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
The number of observations in the model
}
\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{\dontshow{
setParallel(FALSE) # for the CRAN check
}
df <- nelder(~(cl(4)*t(5)) > ind(5))
df$int <- 0
df[df$cl <= 2, 'int'] <- 1
mf1 <- MeanFunction$new(formula = ~ int ,
                        data=df,
                        parameters = c(-1,1)
                        )
mf1$n()
}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-MeanFunction-new"></a>}}
\if{latex}{\out{\hypertarget{method-MeanFunction-new}{}}}
\subsection{Method \code{new()}}{
Create a new MeanFunction object
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{MeanFunction$new(
  formula,
  data,
  parameters = NULL,
  offset = NULL,
  verbose = FALSE
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{formula}}{A \link[stats]{formula} object that describes the mean function, see Details}

\item{\code{data}}{(Optional) A data frame containing the covariates in the model, named in the model formula}

\item{\code{parameters}}{(Optional) A vector with the values of the parameters \eqn{\beta} to use in data simulation and covariance calculations.
If the parameters are not specified then they are initialised to 0.}

\item{\code{offset}}{A vector of offset values (optional)}

\item{\code{verbose}}{Logical indicating whether to report detailed output}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
A MeanFunction object
}
\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{\dontshow{
setParallel(FALSE) # for the CRAN check
}
df <- nelder(~(cl(4)*t(5)) > ind(5))
df$int <- 0
df[df$cl <= 2, 'int'] <- 1
mf1 <- MeanFunction$new(formula = ~ int ,
                        data=df,
                        parameters = c(-1,1),
                        )
}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-MeanFunction-print"></a>}}
\if{latex}{\out{\hypertarget{method-MeanFunction-print}{}}}
\subsection{Method \code{print()}}{
Prints details about the object
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{MeanFunction$print()}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{...}}{ignored}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-MeanFunction-update_parameters"></a>}}
\if{latex}{\out{\hypertarget{method-MeanFunction-update_parameters}{}}}
\subsection{Method \code{update_parameters()}}{
Updates the model parameters
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{MeanFunction$update_parameters(parameters)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{parameters}}{A vector of parameters for the mean function.}

\item{\code{verbose}}{Logical indicating whether to provide more detailed feedback}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-MeanFunction-colnames"></a>}}
\if{latex}{\out{\hypertarget{method-MeanFunction-colnames}{}}}
\subsection{Method \code{colnames()}}{
Returns or replaces the column names of the data in the object
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{MeanFunction$colnames(names = NULL)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{names}}{If NULL then the function prints the column names, if a vector of names, then it attempts to 
replace the current column names of the data}
}
\if{html}{\out{</div>}}
}
\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{\dontshow{
setParallel(FALSE) # for the CRAN check
}
df <- nelder(~(cl(4)*t(5)) > ind(5))
df$int <- 0
df[df$cl <= 5, 'int'] <- 1
mf1 <- MeanFunction$new(formula = ~ int ,
                        data=df,
                        parameters = c(-1,1)
                        )
mf1$colnames(c("cluster","time","individual","treatment"))
mf1$colnames()
}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-MeanFunction-subset_rows"></a>}}
\if{latex}{\out{\hypertarget{method-MeanFunction-subset_rows}{}}}
\subsection{Method \code{subset_rows()}}{
Keeps a subset of the data and removes the rest

All indices not in the provided vector of row numbers will be removed from both the data and fixed effects 
design matrix X.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{MeanFunction$subset_rows(index)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{index}}{Rows of the data to keep}
}
\if{html}{\out{</div>}}
}
\subsection{Returns}{
NULL
}
\subsection{Examples}{
\if{html}{\out{<div class="r example copy">}}
\preformatted{\dontshow{
setParallel(FALSE) # for the CRAN check
}
df <- nelder(~(cl(4)*t(5)) > ind(5))
df$int <- 0
df[df$cl <= 5, 'int'] <- 1
mf1 <- MeanFunction$new(formula = ~ int ,
                        data=df,
                        parameters = c(-1,1)
                        )
mf1$subset_rows(1:20) 
}
\if{html}{\out{</div>}}

}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-MeanFunction-linear_predictor"></a>}}
\if{latex}{\out{\hypertarget{method-MeanFunction-linear_predictor}{}}}
\subsection{Method \code{linear_predictor()}}{
Returns the linear predictor 

Returns the linear predictor, X * beta
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{MeanFunction$linear_predictor()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
A vector
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-MeanFunction-any_nonlinear"></a>}}
\if{latex}{\out{\hypertarget{method-MeanFunction-any_nonlinear}{}}}
\subsection{Method \code{any_nonlinear()}}{
Returns a logical indicating whether the mean function contains non-linear functions of model parameters.
Mainly used internally.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{MeanFunction$any_nonlinear()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
None. Called for effects
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-MeanFunction-clone"></a>}}
\if{latex}{\out{\hypertarget{method-MeanFunction-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{MeanFunction$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}
