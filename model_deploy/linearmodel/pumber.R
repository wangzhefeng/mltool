# plumber.R

#' Echo the parameter that was sent in
#' @param msg The message to echo back.
#' @get /predict
function(v1, v2, v3, v4){
    predict(fit, type = 'response', newdata = data.frame(Sepal.Length = as.numeric(v1), 
                                                         Sepal.Width = as.numeric(v2), 
                                                         Petal.Length = as.numeric(v3), 
                                                         Petal.Width = as.numeric(v4)))
}