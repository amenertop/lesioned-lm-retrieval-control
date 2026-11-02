# Figure 4 in R (ggplot2 + patchwork), plain journal idiom (user, 2026-10-30: all four data figures move from matplotlib
# to R). Same data as fig4.py: estimates through ledger(id), item-level files through fd(); the drawn values are tied to
# the ledger by the assertions below. One conventional chart type per panel, no chain arrows, no badge strips.
#   A stacked horizontal bars of the three error types, patients (10 SD, 8 SA) and the model (intact, WR8, SC4)
#   B bars with intervals: change in miscue cost after SC4 lesions at the prompt positions named on the x axis
#   C per-concept scatter of WR8 naming loss against cue-dependence change, and the mean changes as dots and intervals
#   D dumbbell of cue rescue, one thin row per patient and one thick row per pooled group and model state
#   "D:/R/R-4.6.0/bin/x64/Rscript.exe" docs/paper-prep/figures/r/fig4.R
# Output: manuscript/figures/Figure_4.{pdf,png,svg,tiff}, Figure_4_gray.png; QA manifest and alignment report under
# docs/paper-prep/figures/qa/.

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])), "figkit.R"))
QA <- file.path(ROOT, "docs/paper-prep/figures/qa"); dir.create(QA, showWarnings = FALSE, recursive = TRUE)
# figkit's ledger() renames the first column on a local copy only (names(.ledger)[1] <- "id" after the <<-), so every
# call after the first fails with "Unknown column id"; load the ledger here with the id column in place (kit fix proposed)
.ledger <- read_csv(file.path(ROOT, "docs/paper-prep/results_ledger.csv"), show_col_types = FALSE,
                    col_types = cols(.default = col_character()), locale = locale(encoding = "UTF-8"))
names(.ledger)[1] <- "id"
W_MM <- 180; H_MM <- 128

# ---- helpers (candidates for figkit.R)
lighter <- function(col, t) {            # mix a colour with white: t = 0 keeps it, t = 1 is white
  r <- grDevices::col2rgb(col)[, 1] / 255
  grDevices::rgb(r[1] + (1 - r[1]) * t, r[2] + (1 - r[2]) * t, r[3] + (1 - r[3]) * t)
}
ledger_nature <- function(id) { ledger(id); .ledger[[10]][.ledger$id == id] }     # column 性质 (evidence tier)
confirmatory <- function(id) startsWith(ledger_nature(id), "\u786e\u8ba4\u6027")   # 确认性 → filled symbol
hatch_segments <- function(x0, x1, y0, y1, slope = 38, step = 0.013) {
  # diagonal hatching of a data-space rectangle as plain segments cut at its edges (no PDF pattern, so the collision
  # audit and svglite see ordinary lines); slope and step are in data units, chosen for about 45 degrees on the page
  cs <- seq(x0 - (y1 - y0) / slope, x1, by = step)
  out <- lapply(cs, function(c) {
    lo <- max(x0, c); hi <- min(x1, c + (y1 - y0) / slope)
    if (hi - lo > 1e-9) data.frame(x = lo, xend = hi, y = y0 + slope * (lo - c), yend = y0 + slope * (hi - c)) else NULL
  })
  do.call(rbind, out)
}
write_panel_manifest <- function(fig, path, width_mm, height_mm, panel_ids, row_groups = NULL, column_groups = NULL,
                                 exemptions = list()) {
  # the backend-neutral manifest of nature-figure/scripts/panel_alignment.R, probed on cairo_pdf so Arial's metrics
  # (not Helvetica's from the base pdf device) size the axis cells; audited by audit_panel_alignment.py
  probe <- tempfile(fileext = ".pdf")
  grDevices::cairo_pdf(probe, width = width_mm / 25.4, height = height_mm / 25.4)   # open before the grob is built, so
  on.exit({ grDevices::dev.off(); unlink(probe) }, add = TRUE)                       # no default pdf() device measures text
  grob <- patchwork::patchworkGrob(fig)
  grid::grid.newpage(); grid::grid.draw(grob); grid::grid.force()
  rows <- grob$layout[grepl("^panel(?:-[0-9]+)?$", grob$layout$name, perl = TRUE), , drop = FALSE]
  rows <- rows[order(rows$t, rows$l), , drop = FALSE]
  stopifnot(nrow(rows) == length(panel_ids))
  resolve <- function(u, total, conv) {   # absolute cells convert directly; null cells share what is left, in proportion
    isnull <- grid::unitType(u) == "null"; out <- numeric(length(u))
    if (any(!isnull)) out[!isnull] <- conv(u[!isnull], "bigpts", valueOnly = TRUE)
    if (any(isnull)) { nv <- as.numeric(u[isnull]); out[isnull] <- (total - sum(out)) * nv / sum(nv) }
    out
  }
  W <- width_mm / 25.4 * 72; H <- height_mm / 25.4 * 72
  w <- resolve(grob$widths, W, grid::convertWidth); h <- resolve(grob$heights, H, grid::convertHeight)
  stopifnot(abs(sum(w) - W) < 1e-6, abs(sum(h) - H) < 1e-6)
  xe <- c(0, cumsum(w)); te <- c(0, cumsum(h))
  panels <- lapply(seq_len(nrow(rows)), function(i) {
    r <- rows[i, ]
    list(id = panel_ids[i], bbox_pt = c(xe[r$l], H - te[r$b + 1], xe[r$r + 1], H - te[r$t]), grid_id = "patchwork-grid-1",
         row_start = r$t - 1L, row_stop = r$b, col_start = r$l - 1L, col_stop = r$r)
  })
  m <- list(schema_version = 1L, backend = "r-patchwork", figure = list(width_pt = W, height_pt = H),
            panels = panels, exemptions = exemptions)
  if (!is.null(row_groups)) m$row_groups <- row_groups
  if (!is.null(column_groups)) m$column_groups <- column_groups
  jsonlite::write_json(m, path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
  invisible(m)
}

# ---- data: A, error composition (ledger IDs as in fig4.py)
PM <- "title_gap_patient_markers-"; CL <- "title_gap_clinical_final_cues-"; E <- "experiment3-"; O <- "optogenetics-"
ERR <- c("coordinate", "superordinate", "associate")
err_ids <- list(
  SD = paste0(PM, c("22b6a4a99d4b", "444a8f678f57", "c6e192f47af1")),
  SA = paste0(PM, c("7cc27b874df4", "7b4e973bdb7e", "80ca3121e20a")),
  intact = paste0(CL, c("a3c0910e7a63", "48ece88dc1a6", "ea90f558bb5d")),
  WR8 = paste0(CL, c("9f0bcc7d477d", "53032686eaad", "71d5025b1782")),
  SC4 = paste0(CL, c("ea30789c8946", "7fb0546538c7", "26492dfc0d78")))
err <- lapply(err_ids, function(ids) setNames(lapply(ids, ledger), ERR))
patients <- fd("items_fig4a_patient_errors.csv") %>% select(patient_id, group, error_type, share) %>%
  pivot_wider(names_from = error_type, values_from = share)
for (g in c("SD", "SA")) for (t in ERR) {       # the per-patient shares reproduce the ledger group means
  v <- patients[[t]][patients$group == g]
  stopifnot(abs(mean(v) - err[[g]][[t]]$est) < 1e-9, length(v) == as.integer(err[[g]][[t]]$n))
}
COL <- c(SD = C_SD, SA = C_SA, intact = C_GRAY, WR8 = C_WR8, SC4 = C_SC4)
rows_a <- NULL; y <- 0
for (g in c("SD", "SA")) {
  p <- patients %>% filter(group == g) %>% arrange(desc(coordinate))
  rows_a <- bind_rows(rows_a, tibble(kind = g, label = p$patient_id, y = y + seq_len(nrow(p)) - 1, h = 0.78,
                                     coordinate = p$coordinate, superordinate = p$superordinate, associate = p$associate))
  y <- y + nrow(p) + 1
}
y <- y + 0.7
for (k in c("intact", "WR8", "SC4")) {
  rows_a <- bind_rows(rows_a, tibble(kind = k, label = k, y = y, h = 1.5, coordinate = err[[k]]$coordinate$est,
                                     superordinate = err[[k]]$superordinate$est, associate = err[[k]]$associate$est))
  y <- y + 1.9
}
rows_a <- rows_a %>% mutate(colour = COL[kind], yy = -y)
rects_a <- rows_a %>% transmute(kind, colour, yy, h, x_coord = coordinate, x_sup = coordinate + superordinate,
                                x_ass = coordinate + superordinate + associate) %>%
  mutate(xmin_c = 0, xmax_c = x_coord, xmin_s = x_coord, xmax_s = x_sup, xmin_a = x_sup, xmax_a = x_ass)
solid_a <- bind_rows(
  rects_a %>% transmute(xmin = xmin_c, xmax = xmax_c, ymin = yy - h / 2, ymax = yy + h / 2, fill = colour),
  rects_a %>% transmute(xmin = xmin_s, xmax = xmax_s, ymin = yy - h / 2, ymax = yy + h / 2,
                        fill = vapply(colour, lighter, "", t = 0.55))) %>% filter(xmax - xmin > 1e-9)
open_a <- rects_a %>% transmute(xmin = xmin_a, xmax = xmax_a, ymin = yy - h / 2, ymax = yy + h / 2, colour) %>%
  filter(xmax - xmin > 1e-9)
hatch_a <- bind_rows(lapply(seq_len(nrow(open_a)), function(i) {
  r <- open_a[i, ]; cbind(hatch_segments(r$xmin, r$xmax, r$ymin, r$ymax), colour = unname(r$colour))
}))
# key to the three shades, one row above the bars, in a neutral gray (the shade belongs to each group's own colour)
KEY_Y <- 1.5; SW <- 0.045; SH <- 0.62; KEY_COL <- "#555555"
key_a <- tibble(x = c(0, 0.33, 0.70), label = ERR)
key_solid <- tibble(xmin = key_a$x[1:2], xmax = key_a$x[1:2] + SW, ymin = KEY_Y - SH / 2, ymax = KEY_Y + SH / 2,
                    fill = c(KEY_COL, lighter(KEY_COL, 0.55)))
key_open <- tibble(xmin = key_a$x[3], xmax = key_a$x[3] + SW, ymin = KEY_Y - SH / 2, ymax = KEY_Y + SH / 2, colour = KEY_COL)
key_hatch <- cbind(hatch_segments(key_open$xmin, key_open$xmax, key_open$ymin, key_open$ymax), colour = KEY_COL)
ylab_a <- rows_a %>% group_by(kind) %>% summarise(yy = mean(yy), .groups = "drop") %>%
  mutate(label = kind) %>% arrange(desc(yy))

pA <- ggplot() +
  geom_rect(data = solid_a, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill)) +
  geom_rect(data = open_a, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, colour = colour), fill = "white", linewidth = 0.2) +
  geom_segment(data = hatch_a, aes(x = x, xend = xend, y = y, yend = yend, colour = colour), linewidth = 0.2) +
  geom_rect(data = key_solid, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill)) +
  geom_rect(data = key_open, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, colour = colour), fill = "white", linewidth = 0.2) +
  geom_segment(data = key_hatch, aes(x = x, xend = xend, y = y, yend = yend, colour = colour), linewidth = 0.2) +
  geom_text(data = key_a, aes(x = x + SW + 0.015, y = KEY_Y, label = label), hjust = 0, vjust = 0.5, size = 8 / .pt,
            colour = KEY_COL, family = "Arial") +
  scale_fill_identity() + scale_colour_identity() +
  scale_x_continuous("Error share", breaks = c(0, 0.5, 1), labels = c("0", ".5", "1"), limits = c(0, 1.02), expand = c(0, 0)) +
  scale_y_continuous(NULL, breaks = ylab_a$yy, labels = ylab_a$label, limits = c(min(rows_a$yy) - 1, KEY_Y + 1.0), expand = c(0, 0)) +
  labs(title = "Model errs within category", tag = "A") + base +
  theme(axis.ticks.y = element_blank(), axis.line.y = element_blank(), axis.text.y = element_text(size = 8.5))

# ---- data: B, miscue cost change after SC4 lesions by prompt position
hints <- fd("fig4b_hints.csv")
mis <- list(cue = ledger(paste0(O, "7a5283a65d68")),       # R5, SC4 while the cue is read (H4e)
            answer = ledger(paste0(E, "e79966032b54")),    # S3, SC4 at the answer positions (H4d)
            answer2 = ledger(paste0(O, "16fc6b82d061")),   # R5, SC4 at the answer positions (H4e)
            all = ledger(paste0(E, "b0eb8fed8961")))       # SC4 at all positions, report-only (no ledger interval)
h_all <- hints %>% filter(set == "C4", measure == "H")
stopifnot(abs(h_all$estimate - mis$all$est) < 5e-4)        # the figdata interval belongs to the ledger estimate
mis$all$lo <- h_all$ci_low; mis$all$hi <- h_all$ci_high
SEG_X <- c(cue = 1.0, answer = 2.4, all = 3.8)   # the prompt positions where SC4 was lesioned
bars_b <- tibble(key = names(mis), x = c(SEG_X["cue"], SEG_X["answer"] - 0.23, SEG_X["answer"] + 0.23, SEG_X["all"]),
                 w = c(0.62, 0.48, 0.48, 0.62), est = sapply(mis, `[[`, "est"), lo = sapply(mis, `[[`, "lo"),
                 hi = sapply(mis, `[[`, "hi"), filled = sapply(mis, function(v) confirmatory(v$id))) %>%
  mutate(fill = ifelse(filled, C_SC4, "white"))
pB <- ggplot(bars_b) + zero_line() +
  geom_rect(aes(xmin = x - w / 2, xmax = x + w / 2, ymin = 0, ymax = est, fill = fill), colour = C_SC4, linewidth = 0.35) +
  geom_errorbar(aes(x = x, ymin = lo, ymax = hi), width = 0.14, colour = "#333333", linewidth = 0.4) +
  scale_fill_identity() +
  scale_x_continuous("SC4 lesioned at", breaks = SEG_X, labels = c("the cue", "the answer", "all positions"),
                     limits = c(0.3, 4.5), expand = c(0, 0)) +
  scale_y_continuous("Miscue cost change (nat)", breaks = c(-0.8, -0.4, 0, 0.4), limits = c(-0.85, 0.62)) +
  labs(title = "SC4 misleads only at the answer", tag = "B") + base +
  theme(axis.ticks.x = element_blank())

# ---- data: C, cue dependence against WR8 naming loss, and the mean changes
cue_items <- fd("fig4c_cue_vs_loss.csv")
stopifnot(nrow(cue_items) == 100, !anyDuplicated(cue_items$naming_loss_K8), !anyDuplicated(cue_items$dR_K8))
rho <- ledger(paste0(E, "f871a108ce93"))                   # Spearman correlation (H4b)
stopifnot(abs(cor(rank(cue_items$naming_loss_K8), rank(cue_items$dR_K8)) - rho$est) < 5e-5)
h4a <- ledger(paste0(E, "c663dabb3ffe"))                   # SC4 minus WR8 (H4a)
h4c <- ledger(paste0(O, "bddb339a16ae"))                   # WR8 at twice its output (H4c)
parts <- list(WR8_B = ledger(paste0(E, "39b23068c28e")), WR8_H = ledger(paste0(E, "d91fb7bc09ed")),
              SC4_B = ledger(paste0(E, "4043079fd6ea")), SC4_H = ledger(paste0(E, "b0eb8fed8961")))
dep <- c(WR8 = parts$WR8_B$est + parts$WR8_H$est, SC4 = parts$SC4_B$est + parts$SC4_H$est)
stopifnot(abs(dep["WR8"] - mean(cue_items$dR_K8)) < 5e-4, abs((dep["SC4"] - dep["WR8"]) - h4a$est) < 1e-3)
pC <- ggplot(cue_items, aes(naming_loss_K8, dR_K8)) + zero_line() +
  geom_vline(xintercept = 0, linetype = "22", colour = "#999999", linewidth = 0.35) +
  geom_point(colour = C_WR8, shape = 16, size = 0.9, alpha = 0.55) +
  scale_x_continuous("Naming loss (nat)", breaks = c(-5, 0, 5, 10), limits = c(-6.5, 11.5)) +
  scale_y_continuous("Cue dependence (nat)", breaks = c(0, 10, 20), limits = c(-6.5, 24)) +
  labs(title = "Cues help where retrieval fails", tag = "C") + base
means_c <- tibble(set = factor(c("SC4", "WR8", "\u00d72"), c("SC4", "WR8", "\u00d72")),
                  est = c(dep["SC4"], dep["WR8"], h4c$est), lo = c(NA, NA, h4c$lo), hi = c(NA, NA, h4c$hi),
                  colour = c(C_SC4, C_WR8, C_WR8), shape = c(22, 21, 21),
                  fill = c("white", "white", ifelse(confirmatory(h4c$id), C_WR8, "white")))
pC2 <- ggplot(means_c, aes(set, est)) + zero_line() +
  geom_errorbar(data = filter(means_c, !is.na(lo)), aes(ymin = lo, ymax = hi, colour = colour), width = 0.25, linewidth = 0.6) +
  geom_point(aes(colour = colour, shape = shape, fill = fill), size = 2.6, stroke = 1.0) +
  scale_colour_identity() + scale_fill_identity() + scale_shape_identity() +
  scale_x_discrete(NULL, limits = levels(means_c$set)) +    # the one-row errorbar layer would otherwise train ×2 first
  scale_y_continuous("Mean change (nat)", breaks = c(-4, 0, 4), limits = c(-4.7, 4.6)) + base

# ---- data: D, cue rescue of the patients and the model
cue_ids <- list(SD = paste0(PM, c("e6829f2afed2", "26778aefa564")), SA = paste0(PM, c("fdebb8564dd8", "2b1342db3549")),
                intact = paste0(CL, c("b8126b3cee2e", "e700f75eae04")), WR8 = paste0(CL, c("ff8fef4d9966", "916919857795")),
                SC4 = paste0(CL, c("533b1be069db", "87df9c8cfa47")))
cue <- lapply(cue_ids, function(ids) list(two = ledger(ids[1]), fin = ledger(ids[2])))
patient_cues <- fd("items_fig4d_patient_cueing.csv")
for (g in c("SD", "SA")) {                                  # the pooled ratios reproduce the ledger group values
  r <- patient_cues %>% filter(group == g); unc <- sum(r$uncued_correct)
  stopifnot(abs((sum(r$two_correct) - unc) / (30 * nrow(r) - unc) - cue[[g]]$two$est) < 1.5e-3,
            abs((sum(r$final_correct) - unc) / (30 * nrow(r) - unc) - cue[[g]]$fin$est) < 1.5e-3,
            nrow(r) == as.integer(cue[[g]]$two$n))
}
ROWS_D <- c("SD", "SA", "intact", "WR8", "SC4"); YD <- setNames(rev(seq_along(ROWS_D)), ROWS_D)
groups_d <- tibble(kind = ROWS_D, y = YD[ROWS_D], two = sapply(cue, function(v) v$two$est), fin = sapply(cue, function(v) v$fin$est),
                   n = sapply(cue, function(v) v$two$n), colour = COL[ROWS_D], shape = c(23, 23, 21, 21, 22),
                   size = c(2.6, 2.6, 1.7, 2.6, 2.4),
                   fill = ifelse(sapply(cue, function(v) confirmatory(v$two$id)), COL[ROWS_D], "white"))
thin_d <- bind_rows(lapply(c("SD", "SA"), function(g) {
  r <- patient_cues %>% filter(group == g) %>% arrange(rescued_two)
  tibble(two = r$rescued_two, fin = r$rescued_final, y = YD[g] + seq(-0.3, 0.3, length.out = nrow(r)), colour = COL[g])
}))
ends_d <- groups_d %>% select(kind, y, colour, shape, size, fill, two, fin) %>% pivot_longer(c(two, fin), values_to = "x")
pD <- ggplot() +
  geom_segment(data = thin_d, aes(x = two, xend = fin, y = y, yend = y, colour = colour), linewidth = 0.3, alpha = 0.5, lineend = "round") +
  geom_segment(data = groups_d, aes(x = two, xend = fin, y = y, yend = y, colour = colour), linewidth = 1.0, lineend = "butt") +
  geom_point(data = ends_d, aes(x = x, y = y, colour = colour, shape = shape, fill = fill, size = size), stroke = 0.9) +
  geom_text(data = groups_d, aes(x = 1.11, y = y, label = n, colour = colour), hjust = 1, size = 8 / .pt, family = "Arial") +
  annotate("text", x = 1.11, y = 5.62, label = "n", hjust = 1, size = 8 / .pt, fontface = "italic", colour = "#555555",
           family = "Arial") +
  annotate("text", x = 0.01, y = YD["SD"] + 0.62, label = "two letters", hjust = 0, size = 8 / .pt,
           colour = "#555555", family = "Arial") +
  annotate("text", x = cue$SD$fin$est + 0.012, y = YD["SD"] + 0.62, label = "all but last", hjust = 0, size = 8 / .pt,
           colour = "#555555", family = "Arial") +
  scale_colour_identity() + scale_fill_identity() + scale_shape_identity() + scale_size_identity() +
  scale_x_continuous("Share rescued", breaks = c(0, 0.5, 1), labels = c("0", ".5", "1"), limits = c(0, 1.12), expand = c(0, 0),
                     guide = guide_axis(cap = "upper")) +
  scale_y_continuous(NULL, breaks = YD, labels = names(YD), limits = c(0.45, 5.95), expand = c(0, 0)) +
  labs(title = "Cues rescue the model, not SD", tag = "D") + base +
  theme(axis.ticks.y = element_blank(), axis.line.y = element_blank(), axis.text.y = element_text(size = 8.5))

# ---- assemble: A over C with its side axis (left), B over D (right)
design <- "AAAAAABBBBBB\nCCCCEEDDDDDD"
fig <- pA + pB + pC + pD + pC2 +
  plot_layout(design = design, widths = c(rep(10.25, 4), rep(10.5, 2), rep(13, 6)), heights = c(1.12, 1)) &
  theme(plot.background = element_rect(fill = "white", colour = NA))   # no white border stroke under the panel letters

# ---- alignment gate (nature-figure qa-contract, R section): manifest at export size, audited by the Python CLI
manifest_path <- file.path(QA, "Figure_4.panel-layout.json")
m <- write_panel_manifest(fig, manifest_path, W_MM, H_MM, panel_ids = c("a", "b", "c", "c_means", "d"),
  exemptions = list(list(panels = c("c", "c_means", "d"), checks = c("panel-width", "horizontal-gutter"),
    reason = paste("c is the per-concept scatter with a narrower side axis of three means (c_means) and d the",
                   "dumbbell of the whole right column; widths and gaps differ by design, tops and bottoms are shared"))))
bb <- setNames(lapply(m$panels, `[[`, "bbox_pt"), sapply(m$panels, `[[`, "id"))
stopifnot(abs(bb$a[1] - bb$c[1]) < 1.5, abs(bb$a[3] - bb$c_means[3]) < 1.5,   # left column: outer edges shared
          abs(bb$b[1] - bb$d[1]) < 1.5, abs(bb$b[3] - bb$d[3]) < 1.5)          # right column
audit <- "C:/Users/wanghe/.claude/skills/nature-figure/scripts/audit_panel_alignment.py"
status <- system2("python", c(shQuote(audit), shQuote(manifest_path), "--json-out", shQuote(file.path(QA, "Figure_4.alignment.json")),
                              "--overlay-svg", shQuote(file.path(QA, "Figure_4.alignment.svg")),
                              "--tolerance-pt", "1.5", "--gutter-tolerance-pt", "1.5", "--strict"))
if (status != 0) stop("panel alignment gate failed, see ", file.path(QA, "Figure_4.alignment.json"))

export_figure(fig, "Figure_4", W_MM, H_MM)
