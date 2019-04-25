# Logistic Regression Model example
# 对iris二分类

setwd('./linearmodel/')

# data
all_data = iris[iris$Species != 'setosa', ]

# split data
set.seed(1234)
ind = sample(2, nrow(all_data), replace = TRUE, prob = c(0.7, 0.3))
train = all_data[ind == 1, ]
test = all_data[ind == 2, ]

# training model
fit = glm(Species ~ ., family = binomial(link = 'logit'), data = train)

# save model
save(fit, file = "fit.RData")