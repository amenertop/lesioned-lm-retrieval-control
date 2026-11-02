# Figure 2 in R (ggplot2 + patchwork), the plain journal idiom (user, 2026-10-30: all four data figures move from
# matplotlib to R). Same data as the retired fig2.py: item-level files from docs/paper-prep/figdata/ and every estimate
# through ledger(id) from docs/paper-prep/results_ledger.csv. One conventional chart type per panel, no chain arrows,
# no italic verbs, no badge strips, uppercase panel letters, titles of at most six words.
#   "D:/R/R-4.6.0/bin/x64/Rscript.exe" docs/paper-prep/figures/r/fig2.R
# Output: manuscript/figures/Figure_2.{pdf,png,svg,tiff} and Figure_2_gray.png (export_figure in figkit.R); the panel
# layout manifest, the alignment report and its overlay in docs/paper-prep/figures/qa/. The matplotlib outputs stay
# in manuscript/figures/python-v6/. Encoding: figures/conventions.md (WR8 orange circle, SC4 blue square, filled
# symbol = confirmatory test, open = report-only; item-level points small, filled and translucent).

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])), "figkit.R"))

# figkit's ledger() renames the id column on a local copy only (`names(.ledger)[1] <- "id"` inside the function), so
# its second call in a session fails. The table is preloaded once here with the column renamed; proposed fix in
# figkit.R: `names(.ledger)[1] <<- "id"`.
.ledger <- read_csv(file.path(ROOT, "docs/paper-prep/results_ledger.csv"), show_col_types = FALSE,
                    col_types = cols(.default = col_character()), locale = locale(encoding = "UTF-8"))
names(.ledger)[1] <- "id"

# ---- helpers (candidates for figkit.R)
ledger_nature <- function(id) { ledger(id); .ledger[[10]][.ledger$id == id] }          # the ledger's 性质 column
confirmatory <- function(id) startsWith(ledger_nature(id), "\u786e\u8ba4\u6027")      # 确认性...: filled symbol
panel_words <- function(svg) {
  # English words inside the panels: every <text> element of the SVG, tokens with at least one letter (tick numbers,
  # signs and units of punctuation do not count), as figkit.panel_word_count counted them for the matplotlib figures
  txt <- paste(readLines(svg, warn = FALSE, encoding = "UTF-8"), collapse = " ")
  runs <- regmatches(txt, gregexpr("<text\\b[^>]*>.*?</text>", txt, perl = TRUE))[[1]]
  words <- unlist(lapply(runs, function(r) {
    s <- gsub("&amp;", "&", gsub("<[^>]+>", " ", r))
    w <- strsplit(trimws(s), "\\s+")[[1]]
    w[grepl("[A-Za-z]", w)]
  }))
  list(n = length(words), words = words)
}
write_layout_manifest <- function(fig, path, width_mm, height_mm, ids, exemptions = list()) {
  # The backend-neutral panel-layout manifest of the nature-figure skill (scripts/panel_alignment.R), measured here
  # because that helper keeps only cells named "panel-N" and would drop the faceted panel B, whose cell patchwork
  # names "panel; panel-1-1, ...". Each cell is resolved in a layout viewport on a cairo_pdf device of the export
  # size, so null widths and the Arial metrics are those of the exported figure.
  probe <- tempfile(fileext = ".pdf")   # opened first: the grob's text metrics then come from cairo and Arial, not
  grDevices::cairo_pdf(probe, width = width_mm / 25.4, height = height_mm / 25.4)   # from Rscript's default pdf()
  on.exit({ grDevices::dev.off(); unlink(probe) }, add = TRUE)
  g <- patchwork::patchworkGrob(fig)
  rows <- g$layout[grepl("^panel(-[0-9]+)?(;.*)?$", g$layout$name, perl = TRUE), , drop = FALSE]
  rows <- rows[order(rows$t, rows$l), , drop = FALSE]
  stopifnot(nrow(rows) == length(ids))
  grid::grid.newpage()
  grid::pushViewport(grid::viewport(layout = grid::grid.layout(length(g$heights), length(g$widths),
                                                               widths = g$widths, heights = g$heights)))
  panels <- lapply(seq_len(nrow(rows)), function(i) {
    r <- rows[i, ]
    grid::pushViewport(grid::viewport(layout.pos.row = r$t:r$b, layout.pos.col = r$l:r$r))
    lo <- grid::deviceLoc(grid::unit(0, "npc"), grid::unit(0, "npc"), valueOnly = TRUE)
    hi <- grid::deviceLoc(grid::unit(1, "npc"), grid::unit(1, "npc"), valueOnly = TRUE)
    grid::upViewport()
    list(id = ids[i], bbox_pt = 72 * c(lo$x, lo$y, hi$x, hi$y), grid_id = "patchwork-grid-1",
         row_start = r$t - 1L, row_stop = r$b, col_start = r$l - 1L, col_stop = r$r)
  })
  manifest <- list(schema_version = 1L, backend = "r-patchwork",
                   figure = list(width_pt = width_mm / 25.4 * 72, height_pt = height_mm / 25.4 * 72),
                   panels = panels, exemptions = exemptions)
  jsonlite::write_json(manifest, path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
  invisible(manifest)
}

# ---- A: screening scores of all 7,680 components by layer, the WR8 and SC4 members marked
screen <- fd("items_fig2a_screening.csv") %>%
  mutate(f = pmax(frequency_score, 0), c = pmax(control_score, 0)) %>%
  mutate(f = f / max(f), c = -c / max(c))
stopifnot(nrow(screen) == 7680, sum(screen$set %in% "WR8") == 8, sum(screen$set %in% "SC4") == 4)
set.seed(1)
long_a <- screen %>% select(component, layer, set, f, c) %>%
  pivot_longer(c(f, c), names_to = "marker", values_to = "y") %>%
  mutate(x = layer + runif(n(), -0.38, 0.38), member = ifelse(is.na(set) | set == "", "none", set))
members_a <- long_a %>% filter((member == "WR8" & marker == "f") | (member == "SC4" & marker == "c"))
pA <- ggplot() +
  geom_hline(yintercept = 0, colour = "#BBBBBB", linewidth = 0.3) +
  geom_point(data = filter(long_a, member == "none"), aes(x, y), colour = C_LIGHT, size = 0.25, alpha = 0.6, shape = 16) +
  geom_point(data = filter(members_a, member == "WR8"), aes(x, y), colour = C_WR8, shape = SHAPE[["WR8"]], fill = "white",
             size = 2.1, stroke = 0.9) +
  geom_point(data = filter(members_a, member == "SC4"), aes(x, y), colour = C_SC4, shape = SHAPE[["SC4"]], fill = "white",
             size = 2.1, stroke = 0.9) +
  annotate("text", x = 35.8, y = 0.9, label = "WR8", colour = C_WR8, size = 3.0, fontface = "bold", hjust = 1) +
  annotate("text", x = 29.2, y = -0.9, label = "SC4", colour = C_SC4, size = 3.0, fontface = "bold", hjust = 0) +
  scale_x_continuous("Layer", breaks = c(0, 10, 20, 30, 40, 47), expand = expansion(add = 0.8)) +
  scale_y_continuous(NULL, breaks = c(-0.55, 0.55), labels = c("control", "frequency"), limits = c(-1.05, 1.05)) +
  labs(title = "Two sets in different layers", tag = "A") + base +
  theme(axis.ticks.y = element_blank(), axis.line.y = element_blank())

# ---- B: each lesion moves only its marker (dot and interval in two facets, the 200 random sets as a violin)
MARKERS <- c("Frequency effect (nat)", "Control effect (logit)")
SETS <- c("random\nsets", "SC4", "WR8")
rand <- fd("items_fig2b_random_sets.csv") %>%
  mutate(marker = factor(ifelse(marker == "frequency", MARKERS[1], MARKERS[2]), MARKERS), set = factor(SETS[1], SETS))
rf <- rand$effect[rand$marker == MARKERS[1]]
rc <- rand$effect[rand$marker == MARKERS[2]]
stopifnot(length(rf) == 200, length(rc) == 200,
          abs(median(rf) - ledger("knowledge-f8dda0adcc71")$est) < 1e-9,   # the drawn random sets reproduce the ledger
          abs(max(rf) - ledger("knowledge-8c54fa5ee735")$est) < 1e-9,
          abs(median(rc) - ledger("control-f8dda0adcc71")$est) < 1e-9,
          abs(max(rc) - ledger("control-8c54fa5ee735")$est) < 1e-9)
led_b <- bind_rows(
  ledger_row("knowledge-f3ce21647c2f", set = "WR8", marker = MARKERS[1]),   # WR8 frequency effect, 240 concepts
  ledger_row("knowledge-e06843c7270c", set = "SC4", marker = MARKERS[1]),   # SC4 frequency effect
  ledger_row("control-b9d87e63d005",   set = "SC4", marker = MARKERS[2]),   # SC4 control effect, 63 items (H1a)
  ledger_row("knowledge-49ccbda0bfa5", set = "WR8", marker = MARKERS[2])) %>%   # WR8 control effect
  mutate(set = factor(set, SETS), marker = factor(marker, MARKERS),
         colour = ifelse(set == "WR8", C_WR8, C_SC4),
         fill = ifelse(sapply(id, confirmatory), colour, "white"))   # filled only where the ledger says confirmatory
stopifnot(sum(rf >= led_b$est[led_b$id == "knowledge-f3ce21647c2f"]) == ledger("knowledge-f733221fee68")$est,
          sum(rc >= led_b$est[led_b$id == "control-b9d87e63d005"]) == ledger("control-f733221fee68")$est)
pB <- ggplot() + zero_line() +
  geom_violin(data = rand, aes(set, effect), fill = C_GRAY_FILL, colour = NA, width = 0.7, adjust = 1.2) +
  geom_point(data = rand, aes(set, effect), position = position_jitter(width = 0.12, height = 0, seed = 1),
             size = 0.5, colour = C_GRAY, alpha = 0.7, shape = 16) +
  geom_errorbar(data = led_b, aes(set, ymin = lo, ymax = hi, colour = colour), width = 0.18, linewidth = 0.7) +
  geom_point(data = led_b, aes(set, est, colour = colour, fill = fill, shape = set), size = 2.8, stroke = 1.1) +
  scale_colour_identity() + scale_fill_identity() +
  scale_shape_manual(values = c("random\nsets" = 16, SC4 = SHAPE[["SC4"]], WR8 = SHAPE[["WR8"]])) +
  facet_wrap(~marker, scales = "free_y", strip.position = "left") +
  scale_x_discrete(NULL, expand = expansion(add = 0.5)) + scale_y_continuous(NULL) +
  labs(title = "Each lesion moves only its marker", tag = "B") + base +
  theme(panel.spacing.x = unit(4, "mm"))

# ---- C: similar distractors hurt (per item scatter and the fitted line from the figdata file, no band)
sim <- fd("fig2d_similarity.csv") %>% mutate(fit = fitted_damage_C4 - (damage_C4 - task_adjusted_damage_C4))
slope <- ledger("control-0464bf41ef23")   # SC4 loss per SD of distractor similarity, the slope of the drawn line
stopifnot(nrow(sim) == as.integer(slope$n),
          abs(coef(lm(fit ~ standardized_similarity, sim))[[2]] - slope$est) < 1e-9)
pC <- ggplot(sim, aes(standardized_similarity, task_adjusted_damage_C4)) + zero_line() +
  geom_point(colour = C_SC4, shape = 15, size = 1.6, alpha = 0.75) +
  geom_line(aes(y = fit), colour = C_SC4, linewidth = 0.9) +
  scale_x_continuous("Distractor similarity (z)", breaks = -1:3) +
  scale_y_continuous("SC4 loss (logit)", breaks = seq(-10, 10, 5)) + expand_limits(y = c(-10, 10)) +
  labs(title = "Distractors hurt", tag = "C") + base

# ---- D: rare words lose (per concept WR8 loss by frequency group, the ledger difference on a side axis from zero)
con <- fd("items_fig2d_wr8_concepts.csv") %>%
  mutate(group = factor(ifelse(rare == 1, "rare", "common"), c("common", "rare")))
diff <- ledger("knowledge-f3ce21647c2f")   # WR8 frequency effect: mean loss of the rare minus the common concepts
stopifnot(nrow(con) == as.integer(diff$n), sum(con$rare == 1) == 120,
          abs((mean(con$loss_WR8[con$rare == 1]) - mean(con$loss_WR8[con$rare == 0])) - diff$est) < 1e-9)
pD <- ggplot(con, aes(group, loss_WR8)) + zero_line() +
  geom_violin(fill = C_WR8_FILL, colour = NA, width = 0.8, adjust = 1.1) +
  geom_point(position = position_jitter(width = 0.13, height = 0, seed = 1), size = 0.55, colour = C_WR8, alpha = 0.6,
             shape = 16) +
  stat_summary(fun = median, fun.min = function(v) quantile(v, .25), fun.max = function(v) quantile(v, .75),
               geom = "pointrange", colour = C_INK, size = 0.25, linewidth = 0.6, shape = 15) +
  scale_x_discrete(NULL) +
  scale_y_continuous("WR8 loss (nat)", breaks = seq(-10, 15, 5)) + expand_limits(y = c(-10, 15)) +
  labs(title = "Rare words lose", tag = "D") + base
pD2 <- ggplot(ledger_row(diff$id), aes(x = 1, y = est)) + zero_line() +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.3, colour = C_WR8, linewidth = 0.7) +
  geom_point(shape = SHAPE[["WR8"]], fill = ifelse(confirmatory(diff$id), C_WR8, "white"), colour = C_WR8, size = 2.8,
             stroke = 1.1) +
  scale_x_continuous(NULL, breaks = 1, labels = "rare \u2212 common", limits = c(0.4, 1.6)) +
  scale_y_continuous("Difference (nat)", position = "right", limits = c(0, NA), expand = expansion(mult = c(0.02, 0.08))) +
  base + theme(axis.line.x = element_blank(), axis.ticks.x = element_blank())

# ---- assemble: A across the top; B (two facets), C, D and its difference axis in one row. B is wide enough for its
# three tick labels per facet; D is narrowed so that D and its difference axis together take about the width of C.
# The plot backgrounds lose their white stroke, which the collision audit otherwise reads as a path crossing the
# panel letters.
fig <- pA + pB + pC + pD + pD2 +
  plot_layout(design = "AAAA\nBCDE", widths = c(2.75, 1.2, 0.9, 0.4), heights = c(1, 2.3)) &
  theme(plot.background = element_rect(fill = "white", colour = NA))

# ---- alignment gate (nature-figure skill, 1.5 pt), run before the export so a failing layout overwrites nothing
QA <- file.path(ROOT, "docs/paper-prep/figures/qa"); dir.create(QA, showWarnings = FALSE, recursive = TRUE)
manifest <- file.path(QA, "Figure_2.panel-layout.json")
write_layout_manifest(fig, manifest, 190, 118, ids = c("a", "b", "c", "d", "d_diff"), exemptions = list(
  list(panels = I("b"), checks = I("panel-width"),
       reason = "b holds two facets side by side (frequency effect and control effect), so its plot area is wider than the single panels c and d by design; its top, bottom, left edge and gutter to c are checked"),
  list(panels = I("d"), checks = I("panel-width"),
       reason = "d is narrowed so that d and its difference axis d_diff together take about the width of c, as in the matplotlib version (c and d shared the left edge, not the width); its top, bottom and gutters are checked"),
  list(panels = I("d_diff"), checks = c("panel-width", "horizontal-gutter"),
       reason = "the difference axis of d is a narrow strip (one estimate, axis on the right, no left axis) beside d, as the matplotlib version excluded its axes; its top and bottom are checked with the row")))
AUDIT <- file.path(Sys.getenv("USERPROFILE"), ".claude/skills/nature-figure/scripts/audit_panel_alignment.py")
PY <- "C:/Users/wanghe/.conda/envs/lesion/python.exe"
if (file.exists(AUDIT)) {
  out <- suppressWarnings(system2(PY, c(shQuote(AUDIT), shQuote(manifest), "--json-out",
                                        shQuote(file.path(QA, "Figure_2.alignment.json")), "--overlay-svg",
                                        shQuote(file.path(QA, "Figure_2.alignment.svg")), "--tolerance-pt", "1.5",
                                        "--gutter-tolerance-pt", "1.5", "--strict"), stdout = TRUE, stderr = TRUE))
  cat(out, sep = "\n")
  if (!is.null(attr(out, "status")) && attr(out, "status") != 0) stop("panel alignment gate failed")
} else message("alignment audit script not found, manifest written only: ", manifest)

# ---- export (PDF, PNG 600 dpi, SVG, TIFF, grayscale PNG) and the word count inside the panels
export_figure(fig, "Figure_2", 180, 128)
pw <- panel_words(file.path(ROOT, "manuscript/figures/Figure_2.svg"))
cat("words inside panels:", pw$n, "\n", paste(pw$words, collapse = " "), "\n")
stopifnot(pw$n <= 70)
