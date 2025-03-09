library(mgcv)
library(opera)
library(dplyr)

elecdata <- read.csv("train.csv")

train_idx <- 1 : floor(dim(elecdata)[1] * 0.85)

elecdata$Date <- as.numeric(as.Date(elecdata$Date))
elecdata$WeekDays <- as.factor(elecdata$WeekDays)
elecdata$BH <- as.factor(elecdata$BH)

train <- elecdata[train_idx,]
val <- elecdata[-train_idx,]

gam <- gam(Net_demand ~ s(Date) + s(Temp) + s(Temp_s99) + te(Wind, Wind_weighted) + 
                   te(Nebulosity, Nebulosity_weighted) + s(Load.1) + s(Load.7) + s(Net_demand.7) + 
                   s(toy, k=30) + WeekDays + BH,
                 data = train, gamma=2)

rollkalman <- function(X_pred, y_true, W, theta_0, V0, Q) {
  n <- nrow(X_pred)
  d <- ncol(X_pred)

  thetas <- matrix(nrow = n - W + 1, ncol = d) 
  thetas[1, ] <- as.vector(theta_0) 
  sigmas <- numeric(n - W)
  V <- V0

  for (i in 1:(n - W)) {
    X <- X_pred[i:(i + W - 1), , drop = FALSE]
    y <- y_true[(i + 1):(i + W)] 
    x <- X_pred[i + W,]
    
    V <- V + Q
    Vinv <- solve(V)
    V <- solve(crossprod(X, X) + Vinv)

    theta <- V %*% (crossprod(X, y) + Vinv %*% matrix(thetas[i, ], ncol = 1))
    thetas[i + 1, ] <- theta
    
    resid <- y - X %*% theta
    sigmas[i] <- sqrt(mean(resid^2) * (1 + x %*% V %*% x)) 
  }
  
  thetas <- thetas[-1, , drop = FALSE]
  mus <- rowSums(X_pred[(W + 1):n, , drop = FALSE] * thetas)
  
  return(list(thetas = thetas, mus = mus, sigmas = sigmas))
}

W = 7
common_cols <- intersect(names(train), names(val))
X_des = predict(gam, rbind(train[(nrow(train) - W + 1):nrow(train),][common_cols], val[common_cols]), type="lpmatrix")
y_true = c(train$Net_demand.1[(nrow(train) - W + 1):nrow(train)], val$Net_demand.1)

theta_0 = gam$coefficients

alphas = 10^seq(-5, 0, length.out=5)
betas = 10^seq(-5, 0, length.out=5)

grid <- expand.grid(alpha = alphas, beta = betas)
grid <- as.matrix(grid) 

pred_mus = matrix(nrow=nrow(val), ncol=nrow(grid))
pred_sigmas = matrix(nrow=nrow(val), ncol=nrow(grid))

for (i in 1 : nrow(grid)) {
  print("Itération...")
  alpha <- grid[i, 1]
  beta <- grid[i, 2]
  Alpha <- alpha * diag(length(theta_0))
  Beta <- beta * diag(length(theta_0))
  pred <- rollkalman(X_des, y_true, W, theta_0, Alpha, Beta)
  pred_mus[, i] <- pred$mus
  pred_sigmas[, i] <- pred$sigmas
  print(i)
}

# Changer les paramètre 0.8 et 0.5 en fonction de la loss

experts = pred_mus + qnorm(0.8) * pred_sigmas

aggregation <- oracle(experts[-nrow(experts),], Y = val$Net_demand.1[-1], loss.type = list(name = "pinball", tau = 0.8))

summary(aggregation)

last_pred = experts[nrow(experts),] %*% t(aggregation$coefficients)
aggregated_predictions <- rbind(aggregation$prediction, last_pred)

pinball_loss <- function(y, y_hat, tau = 0.8) {
  error <- y - y_hat
  mean(ifelse(error > 0, tau * error, (tau - 1) * error))
}

pinball_loss5 <- function(y, y_hat, tau = 0.5) {
  error <- y - y_hat
  mean(ifelse(error > 0, tau * error, (tau - 1) * error))
}

final_loss <- pinball_loss(val$Net_demand, aggregated_predictions)
print(final_loss)

# 计算 MAPE
mape <- function(y, y_hat) {
  mean(abs((y - y_hat) / y), na.rm = TRUE) * 100
}

# 计算最终 MAPE
final_mape <- mape(val$Net_demand, aggregated_predictions)
print(final_mape)

final_loss5 <- pinball_loss5(val$Net_demand, aggregated_predictions)
print(final_loss5)

rmse <- sqrt(mean((val$Net_demand - aggregated_predictions)^2))
print(rmse)

plot(aggregated_predictions, type = 'l')
lines(val$Net_demand.1[-1], col = "red")

output <- data.frame(
  Id = seq_len(length(aggregated_predictions)),  
  Net_demand = aggregated_predictions
)

write.table(output, file="submimssion.csv", quote=F, sep=",", dec='.',row.names = F)


