# 01. Loss functions (margin-based)
# 02. Gradients of loss functions (rho.l / rho.m)
# 03. Other functions




##### Packages #####
library(ROCR)




##### 01. Loss functions (margin-based) #####
fn.01 <- function(m) {
  loss <- c()
  loss[m >= 0] <- 0
  loss[m < 0] <- 1
  return(loss)
}
fn.exp <- function(m) {
  loss <- exp(-m)
  return(loss)
}
fn.hinge <- function(m) {
  loss <- sapply(X = m, FUN = function(x) {
    return(max(c(0, 1 - x)))
  })
  return(loss)
}
fn.logit <- function(m) {
  loss <- log(x = 1 + exp(-m), base = 2)
  return(loss)
}
fn.square <- function(m) {
  loss <- (1 - m)^2
  return(loss)
}
fn.savage <- function(m) {
  loss <- 1 / (1 + exp(m))^2
  return(loss)
}
fn.sigmoid <- function(m) {
  loss <- 1 / (1 + exp(m))
  return(loss)
}
fn.bindev <- function(m) {
  loss <- log(x = 1 + exp(-2*m), base = 2)
  return(loss)
}
fn.tan <- function(m) {
  loss <- (2 * atan(m) - 1)^2
  return(loss)
}




##### 02. Gradients of loss functions (rho.l / rho.m) #####
fn.logit.grad <- function(m) {
  a <- 1 / log(x = 2)
  b <- -1 / (1 + exp(m))
  grad <- a * b
  return(grad)
}
fn.exp.grad <- function(m) {
  grad <- - exp(-m)
  return(grad)
}
fn.hinge.grad <- function(m) {
  grad <- c()
  grad[m <= 1] <- -1
  grad[m > 1] <- 0
  return(grad)
}
fn.square.grad <- function(m) {
  grad <- 2 * (m - 1)
  return(grad)
}
fn.savage.grad <- function(m) {
  a <- -2 * exp(m)
  b <- 1 / (1 + exp(m))^3
  grad <- a * b
  return(grad)
}
fn.sigmoid.grad <- function(m) {
  a <- -1 / (1 + exp(-m))
  b <- 1 / (1 + exp(m))
  grad <- a * b
  return(grad)
}
fn.bindev.grad <- function(m) {
  a <- -2 / log(x = 2)
  b <- 1 / (1 + exp(2 * m))
  grad <- a * b
  return(grad)
}
fn.tan.grad <- function(m) {
  a <- 4 * (2 * atan(x = m) - 1)
  b <- 1 / (1 + m^2)
  grad <- a * b
  return(grad)
}




##### 03. Other functions #####
fn.h <- function(w0, w, x) {
  h.x <- w0 + as.numeric(rbind(x) %*% cbind(w)) # for vector/matrix
  return(h.x)
}
fn.init <- function(n) {
  r <- runif(n = n, min = -1, max = 1)
  w0 <- r[1]
  w <- r[2:(n)]
  return(list(w0 = w0, w = w))
}
fn.grad.auc.batch <- function(x, y, ix.batch, w0, w) {
  x.tmp <- x
  grad <- apply(X = rbind(ix.batch), MARGIN = 1, function(x) {
    h <- fn.h(w0 = w0, w = w, x = x.tmp[x,])
    mu <- sum(y[x] * h)
    grad.l.mu <- get(x = paste0("fn.", loss, ".grad"))(mu)
    grad.mu.w0 <- y[x[1]] + y[x[2]]
    grad.mu.w <- colSums(y[x] * x.tmp[x,])
    grad.l.w0 <- grad.l.mu * grad.mu.w0
    grad.l.w <- grad.l.mu * grad.mu.w
    return(c(grad.l.w0, grad.l.w))
  })
  grad <- rowMeans(grad)
  grad <- list(w0 = grad[1], w = grad[-1])
  return(grad)
}
fn.fit.auc <- function(w0, w, x, y, loss, eta, size.grad, itr, itr.print) {
  # n = n.pos + n.neg
  # y = {-1, 1}
  # itr: number of update
  # eta: learning rate
  # loss
  # size.grad
  # Batch: all pairs (size.grad = n.pos * n.neg)
  # Mini-batch: some pairs (size.grad < n.pos)
  # Stochastic: one pair (size.grad = 1)
  
  
  # Input information
  ix.pos <- which(y == 1)
  ix.neg <- which(y == -1)
  n.pos <- length(ix.pos)
  n.neg <- length(ix.neg)
  
  
  # For tracking
  w0.tr <- rep(0, itr + 1)
  w.tr <- matrix(data = 0, nrow = itr + 1, ncol = length(w))
  loss.tr <- rep(0, itr + 1)
  err.tr <- rep(0, itr + 1)
  auc.tr <- rep(0, itr + 1)
  
  
  # Initial values
  w0.tr[1] <- w0
  w.tr[1,] <- w
  h <- fn.h(w0 = w0, w = w, x = x)
  m <- y * h
  loss.tr[1] <- mean(get(x = paste0("fn.", loss))(m))
  err.tr[1] <- mean(get(x = paste0("fn.01"))(m))
  auc.tr[1] <- performance(prediction.obj = prediction(predictions = h, labels = y),
                           measure = "auc")@"y.values"[[1]]
  
  
  # Index of samples for batch in gradient descent method
  if (size.grad == n.pos * n.neg) { # batch
    ix.batch <- as.matrix(expand.grid(ix.pos, ix.neg))
    ix.batch <- array(data = ix.batch, dim = c(n.pos * n.neg, 2, itr))
  } else if (size.grad == 1) { # stochastic
    ix.batch.pos <- sample(x = ix.pos, size = itr, replace = TRUE)
    ix.batch.neg <- sample(x = ix.neg, size = itr, replace = TRUE)
    ix.batch <- rbind(ix.batch.pos, ix.batch.neg)
    ix.batch <- array(data = ix.batch, dim = c(1, 2, itr))
  } else { # mini-batch
    ix.batch <- array(data = 0, dim = c(size.grad, 2, itr))
    for (i in 1:itr) {
      ix.batch.pos <- sample(x = ix.pos, size = size.grad, replace = FALSE)
      ix.batch.neg <- sample(x = ix.neg, size = size.grad, replace = FALSE)
      ix.batch[,,i] <- cbind(ix.batch.pos, ix.batch.neg)
    }
  }
  
  
  # Update
  for (i in 1:itr) {
    w0.old <- w0.tr[i]
    w.old <- w.tr[i,]
    ########################################
    grad <- fn.grad.auc.batch(x = x, y = y, ix.batch = ix.batch[,,i], w0 = w0.old, w = w.old)
    ########################################
    w0.new <- w0.old - (eta * grad$"w0")
    w.new <- w.old - (eta * grad$"w")
    # For tracking
    w0.tr[i + 1] <- w0.new
    w.tr[i + 1,] <- w.new
    h <- fn.h(w0 = w0.new, w = w.new, x = x)
    m <- y * h
    loss.tr[i + 1] <- mean(get(x = paste0("fn.", loss))(m))
    err.tr[i + 1] <- mean(get(x = paste0("fn.01"))(m))
    auc.tr[i + 1] <- performance(prediction.obj = prediction(predictions = h, labels = y),
                                 measure = "auc")@y.values[[1]]
    if (i %% itr.print == 0) {
      cat(paste0(i, " / ", itr, " step, ",
                 "Loss = ", round(x = loss.tr[i + 1], digits = 8), ", ",
                 "AUC = ", round(x = auc.tr[i + 1], digits = 8)), "\n")
    }
  }
  
  
  # Return
  list.trace <- list(loss = loss.tr, err = err.tr, auc = auc.tr, w0 = w0.tr, w = w.tr)
  list.return <- list(w0.fn = w0.tr[i + 1],
                      w.fn = w.tr[i + 1,],
                      trace = list.trace)
  return(list.return)
}