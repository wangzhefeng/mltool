# Rest API -- plumber
# install.packages('plumber', repos = "https://mirrors.tongji.edu.cn/CRAN/")
# install.packages("opencpu")
# library(plumber)

# plumber.R

#' Echo the parameter that was sent in 
#' @param msg The message to echo back.
#' @get /echo

function(msg = ""){
    list(msg = paste0("The message is: '", msg, "'"))
}