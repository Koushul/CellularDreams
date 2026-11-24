suppressPackageStartupMessages({
  library(zeallot); library(dplyr); library(rlang); library(foreach)
  library(MASS); library(ape); library(matrixStats); library(assertthat); library(jsonlite)
})
data <- function(name, envir = parent.frame(), ...) {
  nm <- as.character(substitute(name))
  load(file.path("data", paste0(nm, ".RData")), envir = envir)
}
for (f in c("R/0_opts.R","R/1.1_cif.R","R/1_main.R","R/2_sim.R","R/3.1_spatial.R","R/3.2_dyngrn.R","R/6_technoise.R","R/8_utils.R")) {
  source(f, keep.source = FALSE)
}
dir.create("fixtures", showWarnings = FALSE)

write_f64 <- function(path, x) writeBin(as.double(x), path)

# densities
load("data/param_realdata.zeisel.imputed.RData")
mp <- match_params
mp[, 1:3] <- log10(mp[, 1:3])
dens <- lapply(1:3, function(i) density(mp[, i], n = 2000))
nms <- c("kon", "koff", "s")
for (i in 1:3) {
  write_f64(sprintf("fixtures/den_%s_x.bin", nms[i]), dens[[i]]$x)
  write_f64(sprintf("fixtures/den_%s_y.bin", nms[i]), dens[[i]]$y)
}
load("data/dens_nonzero.RData")
write_f64("fixtures/den_atac_x.bin", dens_nonzero$x)
write_f64("fixtures/den_atac_y.bin", dens_nonzero$y)
cat("dens lens", length(dens[[1]]$x), length(dens_nonzero$x), "\n")

load("data/GRN_params_100.RData")
grn <- GRN_params_100
writeBin(as.integer(nrow(grn)), "fixtures/grn100_n.bin")
writeBin(as.integer(grn[,1]), "fixtures/grn100_tgt.bin")
writeBin(as.integer(grn[,2]), "fixtures/grn100_reg.bin")
write_f64("fixtures/grn100_eff.bin", grn[,3])

# rng references from a fresh seed each time
set.seed(0); write_f64("fixtures/rng_unif.bin", runif(64))
set.seed(0); write_f64("fixtures/rng_norm.bin", rnorm(64))
set.seed(0); write_f64("fixtures/rng_exp.bin", rexp(32))
set.seed(0); write_f64("fixtures/rng_pois.bin", rpois(48, lambda = rep(c(0.3, 2.5, 12.0, 40.0), each = 12)))
set.seed(0); write_f64("fixtures/rng_beta.bin", rbeta(32, shape1 = rep(c(0.2, 0.7, 1.5, 4), each = 8), shape2 = rep(c(0.4, 1.1, 2, 3), each = 8)))
set.seed(0); writeBin(as.integer(sample.int(100000L, 9)), "fixtures/rng_sample9.bin")
set.seed(0); writeBin(as.integer(sample(c(0L,1L), 40, replace = TRUE, prob = c(0.7, 0.3))), "fixtures/rng_sample01.bin")
set.seed(0); writeBin(as.integer(sample(c(0L,1L,2L), 30, replace = TRUE, prob = c(0.1, 0.5, 0.4))), "fixtures/rng_sample012.bin")
set.seed(0); writeBin(as.integer(sample.int(110L, 55)), "fixtures/rng_sample_half.bin")
# weighted large (walker path): 300 bins
set.seed(1); p <- dexp(seq(0.01, 5, length.out = 300)); p <- p/sum(p)
set.seed(2); writeBin(as.integer(sample.int(300L, 40, replace = TRUE, prob = p)), "fixtures/rng_walker.bin")
write_f64("fixtures/rng_walker_p.bin", p)

# seq
write_f64("fixtures/seq_demo.bin", seq(0, 1, by = 1/7))

# trees
set.seed(0)
tr <- Phyla5()
el <- cbind(tr$edge, tr$edge.length)
writeBin(as.integer(nrow(el)), "fixtures/phyla5_n.bin")
writeBin(as.integer(el[,1]), "fixtures/phyla5_from.bin")
writeBin(as.integer(el[,2]), "fixtures/phyla5_to.bin")
write_f64("fixtures/phyla5_len.bin", el[,3])
write_f64("fixtures/phyla5_vcv.bin", as.vector(vcv.phylo(tr, corr = TRUE)))
cat("phyla5 edges\n"); print(el)

set.seed(0)
tr3 <- Phyla3()
el3 <- cbind(tr3$edge, tr3$edge.length)
writeBin(as.integer(nrow(el3)), "fixtures/phyla3_n.bin")
writeBin(as.integer(el3[,1]), "fixtures/phyla3_from.bin")
writeBin(as.integer(el3[,2]), "fixtures/phyla3_to.bin")
write_f64("fixtures/phyla3_len.bin", el3[,3])

# index truncation
x <- c(10, 20, 30, 40)
idx <- c(1, 2.2, 2.9, 3.0)
cat("index trunc", x[idx], "\n")
cat("sample size 2.2\n")
print(try(sample(1:5, size = 2.2)))

# matmul reference
set.seed(0)
A <- matrix(rnorm(8*6), 8, 6)
B <- matrix(rnorm(11*6), 11, 6)
C <- A %*% t(B)
write_f64("fixtures/mm_A.bin", A)
write_f64("fixtures/mm_B.bin", B)
write_f64("fixtures/mm_C.bin", C)

# rank
set.seed(3)
v <- c(1.0, 2.0, 2.0, -1.0, 5.0)
write_f64("fixtures/rank_in.bin", v)
write_f64("fixtures/rank_out.bin", rank(v))

cat("starting beta sim\n")
set.seed(0)
opt <- list(
  GRN = GRN_params_100,
  num.cells = 100,
  num.cifs = 20,
  tree = Phyla5(),
  diff.cif.fraction = 0.8,
  do.velocity = FALSE
)
t0 <- Sys.time()
res <- sim_true_counts(c(opt, list(debug = TRUE)))
cat("beta elapsed", as.numeric(Sys.time()-t0, units="secs"), "\n")
counts <- res$counts
write_f64("fixtures/beta100_counts.bin", counts)
writeBin(as.integer(dim(counts)), "fixtures/beta100_dim.bin")
sel <- c(1:5, 101:105, 10001:10005)
cat("beta selected\n")
print(counts[sel])
write_f64("fixtures/beta100_atac.bin", res$atac_counts)
writeBin(as.integer(dim(res$atac_counts)), "fixtures/beta100_atac_dim.bin")
write_f64("fixtures/beta100_neutral.bin", res$sim$CIF_all$neutral)
cat("neutral dim", paste(dim(res$sim$CIF_all$neutral), collapse="x"), "\n")
cat("done beta\n")

cat("starting velocity sim\n")
set.seed(0)
opt$do.velocity <- TRUE
t0 <- Sys.time()
res <- sim_true_counts(opt)
cat("velocity elapsed", as.numeric(Sys.time()-t0, units="secs"), "\n")
write_f64("fixtures/velo100_counts.bin", res$counts)
write_f64("fixtures/velo100_unspliced.bin", res$unspliced_counts)
write_f64("fixtures/velo100_velocity.bin", res$velocity)
write_f64("fixtures/velo100_atac.bin", res$atac_counts)
cat("done velocity\n")
