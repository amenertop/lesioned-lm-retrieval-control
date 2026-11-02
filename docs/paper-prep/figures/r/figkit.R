# Shared style and helpers for the R figures (ggplot2 + patchwork), 2026-10-30.
# Source this file from docs/paper-prep/figures/r/figN.R:
#   source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])), "figkit.R"))
# It defines ROOT, the colours, the theme `base`, `ledger(id)` for estimates from the results ledger,
# `fd(name)` for item-level files in docs/paper-prep/figdata, and `export_figure()` for the five output formats.
# Conventions (docs/paper-prep/figures/conventions.md): WR8 orange circle, SC4 blue square, joint pink triangle,
# patients black (SD) and dark gray (SA) pentagons, Arial, 190 mm wide, panel letters uppercase, titles of at most
# six words, no legends where direct labels work, one conventional chart type per panel.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(readr); library(dplyr); library(tidyr); library(scales)
})

.kit_file <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NULL)   # set by source()
if (is.null(.kit_file)) .kit_file <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
ROOT <- normalizePath(file.path(dirname(.kit_file), "..", "..", "..", ".."))

# ---- colours
C_WR8 <- "#D55E00"; C_SC4 <- "#0072B2"; C_JOINT <- "#CC79A7"
C_SD <- "#000000"; C_SA <- "#666666"
C_INK <- "#222222"; C_GRAY <- "#8C8C8C"; C_LIGHT <- "#C8C8C8"; C_AXIS <- "#555555"
C_WR8_FILL <- "#F6D9C6"; C_SC4_FILL <- "#CFE3F2"; C_GRAY_FILL <- "#D9D9D9"
SHAPE <- c(WR8 = 21, SC4 = 22, joint = 24, SD = 23, SA = 23, intact = 21)   # hollow shapes with white fill

# ---- theme
base <- theme_classic(base_size = 8.5, base_family = "Arial") +
  theme(axis.line = element_line(linewidth = 0.4, colour = C_AXIS),
        axis.ticks = element_line(linewidth = 0.4, colour = C_AXIS),
        axis.text = element_text(size = 8, colour = "#333333"),
        axis.title = element_text(size = 8.5, colour = C_INK),
        plot.title = element_text(size = 9.5, face = "bold", colour = C_INK, hjust = 0, margin = margin(l = 13, b = 5)),
        plot.title.position = "plot",
        plot.tag = element_text(size = 11, face = "bold", colour = C_INK),
        plot.tag.position = c(0, 1),
        strip.background = element_blank(),
        strip.text = element_text(size = 8.5, colour = C_INK, hjust = 0),
        strip.placement = "outside",
        legend.position = "none",
        plot.margin = margin(4, 6, 2, 4))
zero_line <- function(y = 0) geom_hline(yintercept = y, linetype = "22", colour = "#999999", linewidth = 0.35)

# ---- data
fd <- function(name) read_csv(file.path(ROOT, "docs/paper-prep/figdata", name), show_col_types = FALSE, guess_max = 100000)

.ledger <- NULL
ledger <- function(id) {
  # est, lo, hi from docs/paper-prep/results_ledger.csv (columns: 编号, 结论一句话, 数值, 95%区间下限, 95%区间上限, ...)
  if (is.null(.ledger)) {
    tab <- read_csv(file.path(ROOT, "docs/paper-prep/results_ledger.csv"), show_col_types = FALSE,
                    col_types = cols(.default = col_character()), locale = locale(encoding = "UTF-8"))
    names(tab)[1] <- "id"
    .ledger <<- tab
  }
  r <- .ledger[.ledger$id == id, ]
  if (nrow(r) != 1) stop("ledger id not found once: ", id)
  list(id = id, est = as.numeric(r[[3]]), lo = as.numeric(r[[4]]), hi = as.numeric(r[[5]]), unit = r[[6]], n = r[[7]])
}
ledger_row <- function(id, ...) { l <- ledger(id); tibble(id = id, est = l$est, lo = l$lo, hi = l$hi, ...) }

# ---- export: PDF (cairo, Arial embedded), PNG 600 dpi, SVG, TIFF, grayscale PNG
export_figure <- function(fig, name, width_mm, height_mm, dir = file.path(ROOT, "manuscript/figures")) {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  ggsave(file.path(dir, paste0(name, ".pdf")), fig, width = width_mm, height = height_mm, units = "mm", device = cairo_pdf)
  ggsave(file.path(dir, paste0(name, ".png")), fig, width = width_mm, height = height_mm, units = "mm", dpi = 600, device = ragg::agg_png)
  ggsave(file.path(dir, paste0(name, ".svg")), fig, width = width_mm, height = height_mm, units = "mm", device = svglite::svglite)
  ggsave(file.path(dir, paste0(name, ".tiff")), fig, width = width_mm, height = height_mm, units = "mm", dpi = 600,
         device = ragg::agg_tiff, compression = "lzw")
  py <- "C:/Users/wanghe/.conda/envs/lesion/python.exe"
  # Frontiers asks for RGB TIFFs: semi-transparent layers make ragg write RGBA, so flatten onto white
  tif <- file.path(dir, paste0(name, ".tiff"))
  system2(py, c("-c", shQuote(sprintf(paste0("from PIL import Image; im=Image.open(r'%s'); ",
    "bg=Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.getchannel('A')) if im.mode=='RGBA' else None; ",
    "(bg if im.mode=='RGBA' else im.convert('RGB')).save(r'%s', dpi=(600, 600), compression='tiff_lzw')"), tif, tif))))
  # grayscale copy for the colour-independence check (done with Python's Pillow; R has no magick here)
  src <- file.path(dir, paste0(name, ".png")); dst <- file.path(dir, paste0(name, "_gray.png"))
  system2(py, c("-c", shQuote(sprintf("from PIL import Image; Image.open(r'%s').convert('L').save(r'%s')", src, dst))))
  cat("wrote", file.path(dir, paste0(name, ".pdf")), "\n")
}
