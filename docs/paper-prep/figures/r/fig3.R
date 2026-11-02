# Figure 3 in R (ggplot2 + patchwork), plain journal idiom (user, 2026-10-30: all four data figures move from matplotlib).
# Same data as fig3.py: every estimate through ledger(id) from docs/paper-prep/results_ledger.csv, the item-level
# member contributions from figdata/fig3b_dla.csv, one descriptive interval from figdata/fig4b_hints.csv.
# One conventional chart type per panel, no chain arrows, no italic verbs, no badge strips, no sketch panels.
#   A  dot-and-interval, three facets (naming, six-option choice, word-to-definition matching), WR8 beside SC4
#   B  dot-and-interval, two facets (naming loss, cue benefit change) for the three position-restricted lesions,
#      and a stacked bar of the eight members' direct contributions with the set's interval beside it
#   C  the graded lesions (75, 50, 25, 0% kept) in the plane of naming loss against six-option loss, the joint lesion
#      as a triangle, the intact model at the origin; the sum corner, its dashed lines and the sketches are gone
# Run:  "D:/R/R-4.6.0/bin/x64/Rscript.exe" docs/paper-prep/figures/r/fig3.R      (FIG3_GATE=0 skips the alignment gate)
# Output: manuscript/figures/Figure_3.pdf, .png, .svg, .tiff, _gray.png; QA files in docs/paper-prep/figures/qa/.

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])), "figkit.R"))
QA <- file.path(ROOT, "docs/paper-prep/figures/qa"); dir.create(QA, showWarnings = FALSE, recursive = TRUE)
W_MM <- 180; H_MM <- 128
E <- "experiment3-"; O <- "optogenetics-"
NI_MARGIN <- 0.73   # noninferiority margin for word-to-definition matching (logit), docs/analysis-plan/experiment3.md (S2 rule)

# figkit's ledger() renames the first column to "id" on a local copy only, so its second call fails; preload the
# table here with the renamed column (proposed fix in figkit.R: rename before the `<<-` assignment)
.ledger <- read_csv(file.path(ROOT, "docs/paper-prep/results_ledger.csv"), show_col_types = FALSE,
                    col_types = cols(.default = col_character()), locale = locale(encoding = "UTF-8"))
names(.ledger)[1] <- "id"

# ---- helpers (proposed for figkit.R: ledger_nature(), confirmatory(), val(), with_ci(), tint())
ledger_nature <- function(id) { ledger(id); r <- .ledger[.ledger$id == id, ]; r[[10]] }     # column 性质
confirmatory <- function(id) startsWith(ledger_nature(id), "确认性")                        # filled symbol
val <- function(id) { l <- ledger(id); list(id = id, est = l$est, lo = l$lo, hi = l$hi, conf = confirmatory(id)) }
with_ci <- function(est_id, ci_id) {   # estimate from one ledger row, interval from the O1 curve row of the same run
  e <- ledger(est_id); ci <- ledger(ci_id)
  stopifnot(abs(e$est - ci$est) < 5e-4)
  list(id = est_id, est = e$est, lo = ci$lo, hi = ci$hi, conf = confirmatory(est_id))
}
row_of <- function(v, ...) tibble(est = v$est, lo = v$lo, hi = v$hi, conf = v$conf, ...)
tint <- function(col, t) { x <- col2rgb(col) / 255; rgb(x[1] + (1 - x[1]) * t, x[2] + (1 - x[2]) * t, x[3] + (1 - x[3]) * t) }
fill_key <- function(conf, set) ifelse(conf, as.character(set), "open")
FILLS <- c(WR8 = C_WR8, SC4 = C_SC4, joint = C_JOINT, open = "white")
COLS <- c(WR8 = C_WR8, SC4 = C_SC4, joint = C_JOINT)

# ---- A: the three formats
fmt_lv <- c("Naming\n(nat)", "Six-option\n(logit)", "Matching\n(logit)")   # two lines each: the facets are 18 mm wide
a <- bind_rows(
  row_of(val(paste0(E, "88fae3396929")), set = "WR8", fmt = fmt_lv[1]),                           # S2, confirmatory
  row_of(with_ci(paste0(E, "b212cfe8d967"), paste0(O, "5d5add9cfe69")), set = "SC4", fmt = fmt_lv[1]),
  row_of(with_ci(paste0(E, "ef7425919375"), paste0(O, "6f16be6dd1b8")), set = "WR8", fmt = fmt_lv[2]),
  row_of(with_ci(paste0(E, "9ecb3b2a2794"), paste0(O, "a3782f19a542")), set = "SC4", fmt = fmt_lv[2]),
  row_of(val(paste0(E, "915437d42faa")), set = "WR8", fmt = fmt_lv[3]),                           # S2, confirmatory
  row_of(with_ci(paste0(E, "d707f97abfca"), paste0(O, "67e75de6ae16")), set = "SC4", fmt = fmt_lv[3])) %>%
  mutate(key = fill_key(conf, set), set = factor(set, c("WR8", "SC4")), fmt = factor(fmt, fmt_lv))
# the two S1 contrasts (H2a) are the differences of the drawn single losses
h2a_naming <- ledger(paste0(E, "180de9c20ebb")); h2a_six <- ledger(paste0(E, "e4e7e770013e"))
stopifnot(abs((a$est[1] - a$est[2]) - h2a_naming$est) < 5e-4, abs((a$est[4] - a$est[3]) - h2a_six$est) < 5e-4)
margin_a <- tibble(fmt = factor(fmt_lv[3], fmt_lv), y = NI_MARGIN)
range_a <- tibble(fmt = factor(rep(fmt_lv, each = 2), fmt_lv), set = "WR8",      # axis ranges as in fig3.py
                  est = c(-0.2, 1.3, -1.1, 1.9, -0.3, 0.85))
breaks_a <- function(l) if (diff(l) > 2) c(-1, 0, 1) else c(0, 0.5, 1)           # per facet: six-option, the others
pA <- ggplot(a, aes(set, est)) +
  zero_line() +
  geom_hline(data = margin_a, aes(yintercept = y), linetype = "dotted", colour = "#555555", linewidth = 0.6) +
  geom_text(data = margin_a, aes(x = 1.5, y = y, label = "margin"), vjust = -0.5, size = 8 / .pt, colour = "#555555",
            family = "Arial") +
  geom_blank(data = range_a) +
  geom_errorbar(aes(ymin = lo, ymax = hi, colour = set), width = 0.18, linewidth = 0.7) +
  geom_point(aes(colour = set, shape = set, fill = key), size = 2.8, stroke = 1.1) +
  scale_colour_manual(values = COLS) + scale_fill_manual(values = FILLS) +
  scale_shape_manual(values = SHAPE[c("WR8", "SC4")]) +
  facet_wrap(~fmt, scales = "free_y", strip.position = "top") +
  scale_x_discrete(NULL, expand = expansion(add = 0.5)) +
  scale_y_continuous(NULL, breaks = breaks_a, labels = scales::label_number(drop0trailing = TRUE), expand = expansion(0)) +
  labs(title = "Each set costs its own task", tag = "A") + base +
  theme(panel.spacing.x = unit(4, "mm"), strip.text = element_text(size = 8.5, colour = C_INK, hjust = 0.5))

# ---- B: WR8 lesioned at all positions, at the answer positions, and there with 75% of its output kept
hint <- fd("fig4b_hints.csv") %>% filter(set == "K8", measure == "B")
b_all_cue <- val(paste0(E, "39b23068c28e"))                      # the ledger row stores the estimate without an interval
stopifnot(nrow(hint) == 1, abs(hint$estimate - b_all_cue$est) < 5e-4)
b_all_cue$lo <- hint$ci_low; b_all_cue$hi <- hint$ci_high          # descriptive bootstrap, 100 concepts
row_lv <- c("answer, 75% kept", "answer positions", "all positions")   # bottom to top
meas_lv <- c("Naming loss\n(nat)", "Cue benefit\nchange (nat)")
b <- bind_rows(
  row_of(val(paste0(E, "88fae3396929")), row = row_lv[3], measure = meas_lv[1]),
  row_of(b_all_cue, row = row_lv[3], measure = meas_lv[2]),
  row_of(val(paste0(O, "e18262b1c650")), row = row_lv[2], measure = meas_lv[1]),    # O4
  row_of(val(paste0(O, "8cd2e3a5ae2f")), row = row_lv[2], measure = meas_lv[2]),    # O4, 132 concepts
  row_of(val(paste0(O, "4250de6d812f")), row = row_lv[1], measure = meas_lv[1]),    # O2
  row_of(val(paste0(O, "8946a894574e")), row = row_lv[1], measure = meas_lv[2])) %>%  # O2, 132 concepts
  mutate(set = "WR8", key = fill_key(conf, set), row = factor(row, row_lv), measure = factor(measure, meas_lv))
h2e_naming <- ledger(paste0(O, "9f4957862b96")); h2e_cue <- ledger(paste0(O, "669655b5f7e4"))   # R4 (H2e), caption
stopifnot(abs((b$est[3] - b$est[5]) - h2e_naming$est) < 5e-4)   # naming recovered = answer minus answer with 75% kept
pB <- ggplot(b, aes(est, row)) +
  geom_vline(xintercept = 0, linetype = "22", colour = "#999999", linewidth = 0.35) +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = 0.22, linewidth = 0.7, colour = C_WR8) +
  geom_point(aes(fill = key), shape = SHAPE[["WR8"]], colour = C_WR8, size = 2.8, stroke = 1.1) +
  scale_fill_manual(values = FILLS) +
  facet_wrap(~measure, scales = "free_x", strip.position = "bottom") +
  scale_x_continuous(NULL, breaks = scales::breaks_width(1), expand = expansion(mult = 0.08)) +
  scale_y_discrete(NULL, expand = expansion(add = 0.6)) +
  labs(title = "WR8 acts at the answer", tag = "B") + base +
  theme(panel.spacing.x = unit(5, "mm"), strip.clip = "off",
        strip.text = element_text(size = 8.5, colour = C_INK, hjust = 0.5))

# the stacked bar of the eight members' direct contributions, the set's estimate and interval beside it (S4, H2c)
dla <- fd("fig3b_dla.csv")
c_sum <- val(paste0(E, "d4899b409c4c"))
members <- dla %>% filter(component != "K8_sum") %>% arrange(desc(specific)) %>%
  mutate(component = factor(component, component))
stopifnot(abs(sum(members$specific) - c_sum$est) < 5e-4,                     # members sum to the set's contribution
          abs(dla$specific[dla$component == "K8_sum"] - c_sum$est) < 5e-4)   # the figdata sum row is the ledger row
pos <- members$specific > 0
shade <- setNames(rep("#F2C9AE", nrow(members)), levels(members$component))
shade[pos] <- sapply(seq(0, 0.62, length.out = sum(pos)), function(t) tint(C_WR8, t))
sum_pt <- tibble(x = 1.62, est = c_sum$est, lo = c_sum$lo, hi = c_sum$hi, key = fill_key(c_sum$conf, "WR8"))
pBar <- ggplot() +
  geom_hline(yintercept = 0, colour = "#777777", linewidth = 0.4) +
  geom_col(data = members, aes(x = 1, y = specific, fill = component), width = 0.7, colour = "white", linewidth = 0.3,
           position = position_stack(reverse = TRUE)) +
  geom_errorbar(data = sum_pt, aes(x, ymin = lo, ymax = hi), width = 0.22, linewidth = 0.7, colour = C_WR8) +
  geom_point(data = sum_pt, aes(x, est), shape = SHAPE[["WR8"]], colour = C_WR8, fill = C_WR8, size = 2.8, stroke = 1.1) +
  scale_fill_manual(values = shade) +
  scale_x_continuous(NULL, breaks = 1, labels = "WR8", limits = c(0.5, 2.0)) +
  scale_y_continuous("Direct contribution (logit)", breaks = 0:3, limits = c(-0.08, 3.05), expand = expansion(mult = 0.01)) +
  base + theme(axis.line.x = element_blank(), axis.ticks.x = element_blank())
stopifnot(c_sum$conf)   # drawn filled: S4 is a confirmatory secondary test

# ---- C: the lesion plane, graded lesions and the joint lesion
grade_ids <- tribble(
  ~set,  ~kept, ~naming,        ~six,
  "WR8", 75,    "838983539d83", "d7ef1de56138",
  "WR8", 50,    "d410a5695072", "8d8431b87635",
  "WR8", 25,    "0d05bc447945", "5ce4f5318f12",
  "WR8", 0,     "8c912a82f7fd", "6f16be6dd1b8",
  "SC4", 75,    "8f1416b6746b", "8293b3e7a7cc",
  "SC4", 50,    "32cce154f741", "1a34feaced56",
  "SC4", 25,    "6fff2487b4d3", "1a837c2b2d93",
  "SC4", 0,     "5d5add9cfe69", "a3782f19a542")
grade <- grade_ids %>% rowwise() %>% mutate(
  x = ledger(paste0(O, naming))$est, xlo = ledger(paste0(O, naming))$lo, xhi = ledger(paste0(O, naming))$hi,
  y = ledger(paste0(O, six))$est, ylo = ledger(paste0(O, six))$lo, yhi = ledger(paste0(O, six))$hi,
  conf = confirmatory(paste0(O, naming))) %>% ungroup() %>%
  mutate(key = fill_key(conf, set), complete = kept == 0)
stopifnot(abs(grade$x[grade$set == "WR8" & grade$kept == 0] - a$est[1]) < 5e-4)   # the complete lesions are A's values
h2d_wr8 <- ledger(paste0(O, "64c3430dbf41")); h2d_sc4 <- ledger(paste0(O, "ffc0c31977c1"))   # R1 (H2d), caption
stopifnot(abs(diff(grade$x[grade$set == "WR8" & grade$kept %in% c(75, 0)]) - h2d_wr8$est) < 5e-4,   # 0% minus 75% kept
          abs(diff(grade$y[grade$set == "SC4" & grade$kept %in% c(75, 0)]) - h2d_sc4$est) < 5e-4)
joint <- tibble(set = "joint", x = val(paste0(O, "7191c3e32b10"))$est, xlo = val(paste0(O, "7191c3e32b10"))$lo,
                xhi = val(paste0(O, "7191c3e32b10"))$hi, y = val(paste0(O, "6b1dd3bfcb41"))$est,
                ylo = val(paste0(O, "6b1dd3bfcb41"))$lo, yhi = val(paste0(O, "6b1dd3bfcb41"))$hi,
                key = ifelse(confirmatory(paste0(O, "7191c3e32b10")), "joint", "open"))
inter <- c(naming = ledger(paste0(O, "88d07b667463"))$est, six = ledger(paste0(O, "726cb8916d4a"))$est)   # R6 (H4g)
s_x <- sum(grade$x[grade$kept == 0]); s_y <- sum(grade$y[grade$kept == 0])
stopifnot(abs(joint$x - s_x - inter[["naming"]]) < 1e-3, abs(joint$y - s_y - inter[["six"]]) < 1e-3)   # not drawn
paths <- bind_rows(tibble(set = c("WR8", "SC4"), kept = 100, x = 0, y = 0), grade %>% select(set, kept, x, y)) %>%
  arrange(set, desc(kept))
wr0 <- grade %>% filter(set == "WR8", kept == 0); sc0 <- grade %>% filter(set == "SC4", kept == 0)
wr75 <- grade %>% filter(set == "WR8", kept == 75); sc75 <- grade %>% filter(set == "SC4", kept == 75)
pC <- ggplot() +
  zero_line() + geom_vline(xintercept = 0, linetype = "22", colour = "#999999", linewidth = 0.35) +
  geom_path(data = paths, aes(x, y, colour = set, group = set), linewidth = 0.9) +
  geom_errorbar(data = filter(grade, !complete), aes(x, ymin = ylo, ymax = yhi, colour = set), width = 0, linewidth = 0.5, alpha = 0.45) +
  geom_errorbar(data = filter(grade, !complete), aes(y = y, xmin = xlo, xmax = xhi, colour = set), width = 0, linewidth = 0.5, alpha = 0.45) +
  geom_errorbar(data = filter(grade, complete), aes(x, ymin = ylo, ymax = yhi, colour = set), width = 0.015, linewidth = 0.7, alpha = 0.7) +
  geom_errorbar(data = filter(grade, complete), aes(y = y, xmin = xlo, xmax = xhi, colour = set), width = 0.1, linewidth = 0.7, alpha = 0.7) +
  geom_errorbar(data = joint, aes(x, ymin = ylo, ymax = yhi), colour = C_JOINT, width = 0.015, linewidth = 0.7, alpha = 0.7) +
  geom_errorbar(data = joint, aes(y = y, xmin = xlo, xmax = xhi), colour = C_JOINT, width = 0.1, linewidth = 0.7, alpha = 0.7) +
  geom_point(data = filter(grade, !complete), aes(x, y, colour = set, shape = set, fill = key), size = 1.7, stroke = 0.8) +
  geom_point(data = filter(grade, complete), aes(x, y, colour = set, shape = set, fill = key), size = 2.8, stroke = 1.1) +
  geom_point(data = joint, aes(x, y, fill = key), shape = SHAPE[["joint"]], colour = C_JOINT, size = 3.0, stroke = 1.1) +
  annotate("point", x = 0, y = 0, shape = 21, colour = "#999999", fill = "white", size = 2.2, stroke = 1.0) +
  annotate("text", x = -0.02, y = -0.12, label = "intact", colour = "#999999", size = 2.9, hjust = 1, vjust = 1) +
  annotate("text", x = wr0$x + 0.02, y = wr0$y - 0.16, label = "WR8", colour = C_WR8, size = 3.0, fontface = "bold", hjust = 0, vjust = 1) +
  annotate("text", x = sc0$xhi + 0.02, y = sc0$y + 0.07, label = "SC4", colour = C_SC4, size = 3.0, fontface = "bold", hjust = 0, vjust = 0) +
  annotate("text", x = joint$x - 0.05, y = joint$y + 0.12, label = "joint", colour = C_JOINT, size = 2.9, hjust = 1, vjust = 0) +
  annotate("text", x = wr75$x + 0.02, y = wr75$y + 0.1, label = "75%", colour = C_WR8, size = 2.9, hjust = 0, vjust = 0) +
  annotate("text", x = sc75$x + 0.035, y = sc75$y, label = "75%", colour = C_SC4, size = 2.9, hjust = 0, vjust = 0.5) +
  scale_colour_manual(values = COLS) + scale_fill_manual(values = FILLS) + scale_shape_manual(values = SHAPE[c("WR8", "SC4")]) +
  scale_x_continuous("Naming loss (nat)", breaks = seq(0, 1.25, 0.25), labels = scales::label_number(drop0trailing = TRUE),
                     limits = c(-0.12, 1.3), expand = expansion(0)) +
  scale_y_continuous("Six-option loss (logit)", breaks = -1:1, limits = c(-1.3, 1.95), expand = expansion(0)) +
  labs(title = "The lesions push in two directions", tag = "C") + base

# ---- assemble: A and B (dots, then the bar) across the top, C across the bottom
fig <- pA + pB + pBar + pC +
  plot_layout(design = "ABC\nDDD", widths = c(1.45, 1.1, 0.28), heights = c(1, 1.3)) &
  theme(plot.background = element_blank())   # one white page rectangle below (no per-plot white borders under the letters)
fig <- fig + plot_annotation(theme = theme(plot.background = element_rect(fill = "white", colour = NA)))

# ---- alignment gate (nature-figure skill: panel_alignment.R writes the manifest, audit_panel_alignment.py audits it)
if (Sys.getenv("FIG3_GATE", "1") != "0") {
  skill <- file.path(Sys.getenv("USERPROFILE", path.expand("~")), ".claude/skills/nature-figure/scripts")
  source(file.path(skill, "panel_alignment.R"))
  # the skill's row finder matches "panel-3" only; a facetted plot's cell is named "panel; panel-1-1, ..." in patchwork
  .nature_alignment_panel_rows <- function(grob) {
    rows <- grob$layout[grepl("^panel(-[0-9]+)?($|;)", grob$layout$name, perl = TRUE), , drop = FALSE]
    if (nrow(rows) < 2) stop("At least two patchwork panel cells are required for alignment QA", call. = FALSE)
    rows[order(rows$t, rows$l), , drop = FALSE]
  }
  # the skill's writer probes on the base pdf() device, which has no Arial ("invalid font type"); same function
  # body with a cairo_pdf probe of the export size, so the measured cells are those of the exported PDF
  write_patchwork_panel_layout <- function(plot, manifest_path, width_in, height_in, panel_ids = NULL,
                                           row_groups = NULL, column_groups = NULL, exemptions = list()) {
    probe <- tempfile(fileext = ".pdf")
    grDevices::cairo_pdf(probe, width = width_in, height = height_in)   # open before the grob is built, so the text
    on.exit({ grDevices::dev.off(); unlink(probe) }, add = TRUE)         # metrics are Arial's, not a pdf() fallback
    grob <- patchwork::patchworkGrob(plot)
    panel_rows <- .nature_alignment_panel_rows(grob)
    stopifnot(length(panel_ids) == nrow(panel_rows), !anyDuplicated(panel_ids))
    grid::grid.newpage(); grid::grid.draw(grob); grid::grid.force()
    # the panel cells are "null" units (relative), which convert to 0 pt outside a layout viewport; resolve them as
    # gtable does, the page minus the absolute cells, split by the null weights
    resolve <- function(u, total_pt, conv) {
      types <- grid::unitType(u); pt <- conv(u, "pt", valueOnly = TRUE); nul <- types == "null"; pt[nul] <- 0
      if (any(nul)) pt[nul] <- (total_pt - sum(pt)) * as.numeric(u)[nul] / sum(as.numeric(u)[nul])
      pt
    }
    widths_pt <- resolve(grob$widths, width_in * 72, grid::convertWidth)
    heights_pt <- resolve(grob$heights, height_in * 72, grid::convertHeight)
    x_edges <- c(0, cumsum(widths_pt)); top_edges <- c(0, cumsum(heights_pt)); total_h <- sum(heights_pt)
    panels <- lapply(seq_len(nrow(panel_rows)), function(i) {
      r <- panel_rows[i, ]
      list(id = panel_ids[i], bbox_pt = unname(c(x_edges[r$l], total_h - top_edges[r$b + 1], x_edges[r$r + 1], total_h - top_edges[r$t])),
           grid_id = "patchwork-grid-1", row_start = as.integer(r$t - 1), row_stop = as.integer(r$b),
           col_start = as.integer(r$l - 1), col_stop = as.integer(r$r))
    })
    manifest <- list(schema_version = 1L, backend = "r-patchwork",
                     figure = list(width_pt = width_in * 72, height_pt = height_in * 72), panels = panels, exemptions = exemptions)
    if (!is.null(row_groups)) manifest$row_groups <- .nature_alignment_groups(row_groups)
    if (!is.null(column_groups)) manifest$column_groups <- .nature_alignment_groups(column_groups)
    jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    invisible(manifest)
  }
  rep_align <- require_patchwork_panel_alignment(
    fig, manifest_path = file.path(QA, "Figure_3.alignment-layout.json"),
    report_path = file.path(QA, "Figure_3.alignment.json"), overlay_svg = file.path(QA, "Figure_3.alignment.svg"),
    width_in = W_MM / 25.4, height_in = H_MM / 25.4, panel_ids = c("A", "B", "B_bar", "C"),
    row_groups = list(c("A", "B", "B_bar")),
    exemptions = list(list(panels = c("A", "B", "B_bar"), checks = c("panel-width", "horizontal-gutter"),
                           reason = paste("row 1 holds A (three facets), B (two facets with row labels) and the narrow",
                                          "stacked bar of B; widths and gaps differ by design, tops and bottoms are shared"))),
    audit_script = file.path(skill, "audit_panel_alignment.py"), tolerance_pt = 1.5, gutter_tolerance_pt = 1.5, strict = TRUE)
  cat("alignment:", rep_align$verdict, "\n")
}

export_figure(fig, "Figure_3", W_MM, H_MM)
