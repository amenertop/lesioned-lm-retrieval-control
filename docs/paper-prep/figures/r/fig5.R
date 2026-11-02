# Figure 5 in R (ggplot2 + patchwork), plain journal idiom (user, 2026-10-30: all four data figures move from
# matplotlib to R). Replaces docs/paper-prep/figures/fig5.py, whose panel A (the sketch of what lost knowledge
# predicts) is now one sentence in the caption; the other three panels keep their content and shift one letter:
#   A  "Damage grows, item consistency does not": partial r of naming and definition-choice loss against
#      definition-choice loss for all 32 lesion states (target readout, 120 final families), the prediction of
#      lost knowledge as one light shaded wedge (a sketch, no values), the broad lesion's severity path as a line
#   B  "No diagonal": naming against definition-choice loss per family under five lesions, one facet each,
#      labelled by the lesion's name in words, axes scaled per facet (as fig5.py)
#   C  "Cues still rescue": naming failures rescued by two letters and by all but the last, the patients
#      (thin lines) and their pooled groups as in Fig. 4D, then the intact model, the broad lesion at 10% kept
#      and the learned hub-profile lesion, the number of patients or of uncued failures at the right
# Numbers: every estimate through ledger(id) (docs/paper-prep/results_ledger.csv, IDs as in fig5.py), item-level
# data through fd() (figdata/items_fig5b_families.csv, items_fig5c_badges.csv, items_fig4d_patient_cueing.csv).
# Nothing is recomputed except to check it against the ledger (stopifnot below).
#   "D:/R/R-4.6.0/bin/x64/Rscript.exe" docs/paper-prep/figures/r/fig5.R
# Output: manuscript/figures/Figure_5.{pdf,png,svg,tiff}, Figure_5_gray.png (export_figure), then the alignment
# manifest and report in docs/paper-prep/figures/qa/Figure_5.alignment*.json and the word count on stdout.

source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])), "figkit.R"))
suppressPackageStartupMessages({ library(grid); library(jsonlite) })
# figkit's ledger() renames the first column to "id" on a local copy only (`names(.ledger)[1] <- "id"` inside the
# function), so its second call finds no id column. Load once here and rename the global copy until figkit is fixed.
invisible(ledger("title_gap_fine_core_readout-648ed349c867")); names(.ledger)[1] <- "id"

# ---- helpers (candidates for figkit.R)
pt_size <- function(pt) pt / .pt                      # geom_text size for a type size in points
ledger_field <- function(id, col) { ledger(id); r <- .ledger[.ledger$id == id, ]; r[[col]] }   # 2 claim, 10 tier, 12 metric path
confirmatory <- function(id) startsWith(ledger_field(id, 10), "\u786e\u8ba4\u6027")           # "确认性": filled symbol
C_BROAD <- "#7A7A7A"; C_LEARNED <- "#3F3F3F"; C_MASK <- "#B0B0B0"   # grays of the other lesion kinds (conventions.md)
neg <- function(x) sub("^-", "\u2212", format(x, trim = TRUE))     # tick labels with a real minus sign

# Pentagon markers for the patient groups (no base pch is a pentagon): size is the circumscribed diameter in mm.
GeomPentagon <- ggproto("GeomPentagon", Geom,
  required_aes = c("x", "y"),
  default_aes = aes(colour = "black", fill = "white", size = 2.6, linewidth = 0.4, alpha = NA),
  draw_key = draw_key_point,
  draw_panel = function(data, panel_params, coord) {
    d <- coord$transform(data, panel_params); n <- nrow(d)
    ang <- pi / 2 + 2 * pi * (0:4) / 5; r <- rep(d$size / 2, each = 5)
    grid::polygonGrob(unit(rep(d$x, each = 5), "npc") + unit(r * cos(rep(ang, n)), "mm"),
                      unit(rep(d$y, each = 5), "npc") + unit(r * sin(rep(ang, n)), "mm"),
                      id = rep(seq_len(n), each = 5),
                      gp = grid::gpar(col = d$colour, fill = d$fill, lwd = d$linewidth * .pt, linejoin = "mitre"))
  })
geom_pentagon <- function(mapping = NULL, data = NULL, ..., inherit.aes = TRUE)
  layer(geom = GeomPentagon, mapping = mapping, data = data, stat = "identity", position = "identity",
        params = list(...), inherit.aes = inherit.aes)

# Panel rectangles of a patchwork figure in PDF points, measured by grid after layout on a device of the export
# size (the skill's panel_alignment.R converts the gtable's null units, which come out as 0; this reads the drawn
# viewports instead). `patterns` maps a panel id to a regex on the viewport name.
panel_boxes_pt <- function(fig, width_mm, height_mm, patterns) {
  grDevices::pdf(NULL, width = width_mm / 25.4, height = height_mm / 25.4)   # before the grob is built, or
  on.exit(grDevices::dev.off(), add = TRUE)                                   # text metrics open Rplots.pdf
  g <- patchwork::patchworkGrob(fig)
  grid.newpage(); grid.draw(g); grid.force()
  vps <- grid.ls(viewports = TRUE, grobs = FALSE, print = FALSE)
  nm <- unique(vps$name[vps$type == "vpListing"])
  lapply(patterns, function(p) {
    hit <- grep(p, nm, value = TRUE, perl = TRUE)
    if (length(hit) != 1) stop("viewport pattern matched ", length(hit), " names: ", p)
    downViewport(hit)
    lo <- deviceLoc(unit(0, "npc"), unit(0, "npc"), valueOnly = TRUE)
    hi <- deviceLoc(unit(1, "npc"), unit(1, "npc"), valueOnly = TRUE)
    upViewport(0)
    c(lo$x, lo$y, hi$x, hi$y) * 72       # inches to pt: [left, bottom, right, top]
  })
}
write_alignment_manifest <- function(boxes, grid_pos, width_mm, height_mm, path, exemptions = list()) {
  panels <- lapply(names(boxes), function(id) {
    gp <- grid_pos[[id]]
    list(id = id, bbox_pt = unname(boxes[[id]]), grid_id = gp$grid, row_start = gp$r[1], row_stop = gp$r[2],
         col_start = gp$c[1], col_stop = gp$c[2])
  })
  m <- list(schema_version = 1L, backend = "r-patchwork",
            figure = list(width_pt = width_mm / 25.4 * 72, height_pt = height_mm / 25.4 * 72),
            panels = panels, exemptions = exemptions)
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  jsonlite::write_json(m, path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
  invisible(m)
}
panel_word_count <- function(svg) {   # as figkit.panel_word_count: tokens with a letter in every <text> element
  txt <- paste(readLines(svg, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
  m <- regmatches(txt, gregexpr("<text\\b[^>]*>(.*?)</text>", txt, perl = TRUE))[[1]]
  s <- gsub("<[^>]+>", " ", m); s <- gsub("&amp;", "&", s)
  w <- unlist(strsplit(s, "\\s+")); w <- w[grepl("[A-Za-z]", w)]
  list(n = length(w), words = w)
}

# ---- A: the 32 lesion states (state, kind, definition-choice ID, partial r ID), as the STATES table of fig5.py
FR <- "title_gap_fine_remaining_core-"; FC <- "title_gap_fine_core_readout-"; CO <- "title_gap_coarse_final_core-"
states <- tribble(
  ~state, ~kind, ~did, ~rid,
  "WR8_a0", "WR8", paste0(FR, "72bb93644bd4"), "case_rescore-13052e18cefc",
  "SC4_a0", "SC4", paste0(FR, "3e84cf723965"), "case_rescore-18cbeaab68ee",
  "WR8_a0_SC4_a0", "joint", paste0(FR, "96531c774ac8"), "case_rescore-2188d651a017",
  "WR8_a0_SC4_a0.5", "joint", paste0(FR, "def6594bbdb0"), "case_rescore-6a65bc29d225",
  "WR8_a0_comprehension_s0_learned_hard_64", "joint", paste0(FR, "747d1ebbe6e8"), "case_rescore-6bb8b39e3cb1",
  "broad_middle_f1_a0.9_s0", "broad", paste0(FR, "f3e38771b597"), "case_rescore-4ab6e41876c3",
  "broad_middle_f1_a0.75_s0", "broad", paste0(FR, "d96158617eef"), "case_rescore-5b20640ea078",
  "broad_middle_f1_a0.5_s0", "broad", paste0(FR, "c124f9cef436"), "case_rescore-93e837ffe8ed",
  "broad_middle_f1_a0.25_s0", "broad", paste0(FR, "9f8281ff53f9"), "case_rescore-bc596db0291d",
  "broad_middle_f1_a0.1_s0", "broad", paste0(FR, "f34bb8f3d165"), "case_rescore-6f25626e4b65",
  "broad_early_f1_a0.5_s0", "broad", paste0(FR, "da203c09b819"), "case_rescore-1e0c3e860933",
  "broad_late_f1_a0.75_s0", "broad", paste0(FR, "62852e442bb8"), "case_rescore-2d7a70abc1ef",
  "hub_s2_learned_soft", "learned", paste0(FC, "36b6a70af49b"), "case_rescore-9de58f0685df",
  "comprehension_s2_learned_soft", "learned", paste0(FC, "bc43f20250b2"), "case_rescore-cf8fa9e97d68",
  "relative_hub_s0_learned_soft", "learned", paste0(FC, "25a1b8fcc6b8"), "case_rescore-31adb9f1e9fa",
  "hub_s2_learned_hard_4", "learned", paste0(FC, "d4c7d1877cb4"), "case_rescore-adc3f525db32",
  "comprehension_s0_learned_hard_64", "learned", paste0(FR, "7e36721969ce"), "case_rescore-4fb0f11168c8",
  "hub_s0_learned_hard_64", "learned", paste0(FR, "3e4b28634381"), "case_rescore-4de504e480ef",
  "hub_s1_learned_hard_64", "learned", paste0(FR, "d5d1ccf133b1"), "case_rescore-8b797a487ab0",
  "hub_s2_learned_hard_64", "learned", paste0(FR, "a82ce11b17b2"), "case_rescore-bb04dc08cb9d",
  "relative_hub_s0_learned_hard_64", "learned", paste0(FR, "193be4b1c180"), "case_rescore-048efd29c203",
  "relative_hub_s1_learned_hard_64", "learned", paste0(FR, "838d7438eba3"), "case_rescore-8a960b13497c",
  "relative_hub_s2_learned_hard_64", "learned", paste0(FR, "1a2fdff10ea7"), "case_rescore-7b6cc79ebb9a",
  "weight_q_L4_s0.5_seed1910", "mask", paste0(CO, "9ab5195cb467"), "case_rescore-86b354a7c0dd",
  "weight_q_L4_s0.5_seed1911", "mask", paste0(CO, "4e0e31ba43cb"), "case_rescore-94d9f9856d09",
  "weight_q_L4_s1_seed1910", "mask", paste0(CO, "aec741bc8ac3"), "case_rescore-9772b15195b9",
  "weight_v_L4_s1_seed1910", "mask", paste0(CO, "7a291c22d240"), "case_rescore-0f04461ab842",
  "weight_gate_L28_s1_seed1910", "mask", paste0(CO, "87dadab5420f"), "case_rescore-58bd0496af73",
  "weight_up_L28_s0.5_seed1910", "mask", paste0(CO, "b989df2cc655"), "case_rescore-635149d90d53",
  "weight_up_L28_s0.5_seed1911", "mask", paste0(CO, "51867fc6aca7"), "case_rescore-90cc2d5e58a6",
  "weight_gate_L44_s1_seed1910", "mask", paste0(CO, "ac54260a2bf9"), "case_rescore-dc3aca51b6c5",
  "weight_q_L28_s1_seed1910", "mask", paste0(CO, "07def11c46e8"), "case_rescore-8021f1113426")
pts <- states %>% rowwise() %>% mutate(
  x = ledger(did)$est, r = ledger(rid)$est, lo = ledger(rid)$lo, hi = ledger(rid)$hi,
  filled = confirmatory(rid),
  ok = as.integer(ledger(did)$n) == 120 && as.integer(ledger(rid)$n) == 120 &&
    grepl(state, ledger_field(did, 2), fixed = TRUE) && grepl(state, ledger_field(rid, 2), fixed = TRUE) &&
    !grepl("accepted_vs_coordinate", ledger_field(rid, 12), fixed = TRUE)) %>% ungroup()
stopifnot(nrow(pts) == 32, all(pts$ok))
r_max <- pts[which.max(pts$r), ]
stopifnot(r_max$state == "weight_q_L28_s1_seed1910", round(r_max$r, 2) == 0.28, round(r_max$lo, 2) == 0.10,
          round(r_max$hi, 2) == 0.46)   # the largest estimate quoted in Results 3.5 (names scored in both spellings)
KIND_COL <- c(WR8 = C_WR8, SC4 = C_SC4, joint = C_JOINT, broad = C_BROAD, learned = C_LEARNED, mask = C_MASK)
KIND_SHAPE <- c(WR8 = 21, SC4 = 22, joint = 24, broad = 23, learned = 23, mask = 23)
pts <- pts %>% mutate(col = KIND_COL[kind], shp = KIND_SHAPE[kind], fill = ifelse(filled, col, "white"),
                      size = ifelse(kind == "mask", 1.55, 2.0), lw = ifelse(kind == "mask", 0.3, 0.5),
                      kind = factor(kind, c("mask", "broad", "learned", "joint", "SC4", "WR8"))) %>% arrange(kind)
broad_path <- pts %>% filter(state %in% paste0("broad_middle_f1_a", c("0.9", "0.75", "0.5", "0.25", "0.1"), "_s0")) %>%
  mutate(step = match(state, paste0("broad_middle_f1_a", c("0.9", "0.75", "0.5", "0.25", "0.1"), "_s0"))) %>% arrange(step)
at <- function(s) pts[pts$state == s, ]
wedge <- data.frame(x = c(0, 21.5, 21.5), y = c(0, 0.36, 0.64))   # the prediction, a sketch as in fig5.py
lab <- function(x, y, l, col = C_INK, hjust = 0.5, vjust = 0.5, bold = FALSE, pt = 8)
  annotate("text", x = x, y = y, label = l, colour = col, hjust = hjust, vjust = vjust, size = pt_size(pt),
           fontface = if (bold) "bold" else "plain", family = "Arial")
pA <- ggplot() +
  geom_polygon(data = wedge, aes(x, y), fill = "#E6E6E6", colour = NA) +
  zero_line() +
  geom_path(data = broad_path, aes(x, r), colour = C_BROAD, linewidth = 0.5) +
  geom_errorbar(data = pts, aes(x, ymin = lo, ymax = hi, colour = I(col), linewidth = I(lw)), width = 0) +
  geom_point(data = pts, aes(x, r, colour = I(col), shape = I(shp), fill = I(fill), size = I(size)), stroke = 0.7) +
  lab(14.0, 0.545, "lost knowledge predicts", col = "#555555", hjust = 1) +
  lab(at("WR8_a0")$x, at("WR8_a0")$lo - 0.03, "WR8", col = C_WR8, vjust = 1, bold = TRUE, pt = 8.5) +
  lab(1.3, 0.335, "SC4", col = C_SC4, hjust = 0, bold = TRUE, pt = 8.5) +
  lab(at("WR8_a0_comprehension_s0_learned_hard_64")$x, 0.42, "joint", col = C_JOINT, vjust = 0) +
  lab(7.3, -0.25, "broad", col = C_BROAD) +
  lab(4.7, -0.30, "learned", col = C_LEARNED) +
  lab(1.4, -0.36, "masks", col = "#9A9A9A") +
  lab(at("hub_s2_learned_soft")$x - 0.55, 0.17, "learned hub", col = C_LEARNED, hjust = 1) +
  lab(at("broad_middle_f1_a0.1_s0")$x + 0.55, at("broad_middle_f1_a0.1_s0")$r, "broad 10%", col = C_BROAD, hjust = 0) +
  lab(21.4, -0.14, "learned comprehension", col = C_LEARNED, hjust = 1, vjust = 1) +
  scale_x_continuous("Definition-choice loss (logit)", limits = c(-2.4, 21.5), breaks = c(0, 5, 10, 15, 20),
                     expand = expansion(0)) +
  scale_y_continuous("Partial r", limits = c(-0.46, 0.66), breaks = seq(-0.4, 0.6, 0.2),
                     labels = c("\u2212.4", "\u2212.2", "0", ".2", ".4", ".6"), expand = expansion(0)) +
  labs(title = "Damage grows, item consistency stays weak", tag = "A") + base

# ---- B: five lesions, family by family (figdata, means checked against the ledger as in fig5.py)
fam <- fd("items_fig5b_families.csv")
badges <- fd("items_fig5c_badges.csv")
small <- tribble(   # key in the families file, state, label, kind, naming ID, definition ID, partial r ID
  ~key, ~state, ~label, ~kind, ~nid, ~did, ~rid,
  "WR8", "WR8_a0", "WR8", "WR8", "case_rescore-7e29f8cbc130", paste0(FR, "72bb93644bd4"), "case_rescore-13052e18cefc",
  "SC4", "SC4_a0", "SC4", "SC4", "case_rescore-25a74a29025d", paste0(FR, "3e84cf723965"), "case_rescore-18cbeaab68ee",
  "broad_10", "broad_middle_f1_a0.1_s0", "broad 10%", "broad", "case_rescore-3cc923f6ea05", paste0(FR, "f34bb8f3d165"), "case_rescore-6f25626e4b65",
  "learned_comprehension", "comprehension_s2_learned_soft", "learned\ncompre-\nhension", "learned", "case_rescore-0d9f4803ea71", paste0(FC, "bc43f20250b2"), "case_rescore-cf8fa9e97d68",
  "learned_hub", "hub_s2_learned_soft", "learned\nhub", "learned", "case_rescore-a674fe868384", paste0(FC, "36b6a70af49b"), "case_rescore-9de58f0685df")
stopifnot(setequal(unique(badges$state), small$state), setequal(unique(fam$lesion), small$key),
          all(fam$readout == "target"), all(fam$naming_unit == "nat"), all(fam$definition_unit == "logit"))
for (i in seq_len(nrow(small))) {   # the drawn item means reproduce the ledger (10 checks), as fig5.py asserts
  f <- fam[fam$lesion == small$key[i], ]
  stopifnot(nrow(f) == 120, all(f$state == small$state[i]),
            abs(mean(f$naming_loss) - ledger(small$nid[i])$est) < 1e-9,
            abs(mean(f$definition_loss) - ledger(small$did[i])$est) < 1e-9,
            small$state[i] %in% pts$state, ledger(small$rid[i])$est == at(small$state[i])$r)
}
fam <- fam %>% mutate(label = factor(small$label[match(lesion, small$key)], small$label),
                      kind = small$kind[match(lesion, small$key)],
                      col = KIND_COL[kind], shp = c(WR8 = 16, SC4 = 15, broad = 18, learned = 18)[kind])
lims <- fam %>% group_by(label) %>%   # range with zero, padded 8% (fig5.py)
  summarise(x0 = min(c(naming_loss, 0)), x1 = max(c(naming_loss, 0)),
            y0 = min(c(definition_loss, 0)), y1 = max(c(definition_loss, 0)), .groups = "drop") %>%
  mutate(px = 0.08 * (x1 - x0), py = 0.08 * (y1 - y0)) %>%
  transmute(label, xmin = x0 - px, xmax = x1 + px, ymin = y0 - py, ymax = y1 + py) %>%
  pivot_longer(-label, names_to = c("axis", "end"), names_pattern = "(x|y)(min|max)") %>%
  pivot_wider(names_from = axis, values_from = value)
ticks3 <- function(l) { b <- pretty(l, n = 3); b[b > l[1] & b < l[2]] }   # at most a few round steps, as fig5.py
pB <- ggplot(fam, aes(naming_loss, definition_loss)) +
  zero_line() + geom_vline(xintercept = 0, linetype = "22", colour = "#999999", linewidth = 0.35) +
  geom_blank(data = lims, aes(x, y)) +
  geom_point(aes(colour = I(col), shape = I(shp)), size = 1.15, alpha = 0.7) +
  facet_wrap(~label, nrow = 1, scales = "free") +
  scale_x_continuous("Naming loss (nat)", breaks = ticks3, labels = neg, expand = expansion(0)) +
  scale_y_continuous("Definition-choice loss (logit)", breaks = ticks3, labels = neg, expand = expansion(0)) +
  labs(title = "No diagonal", tag = "B") + base +
  theme(panel.spacing.x = unit(3.5, "mm"),   # the strips sit on their panels; the two learned names take two lines
        strip.text = element_text(size = 8.5, colour = C_INK, hjust = 0, vjust = 0, lineheight = 0.95, margin = margin(b = 2)))

# ---- C: cue rescue in the template of Fig. 4D
PM <- "title_gap_patient_markers-"; CL <- "title_gap_clinical_final_cues-"; CS <- "title_gap_clinical_cue_search-"
rescue <- tribble(
  ~row, ~kind, ~col, ~two_id, ~fin_id,
  "SD", "SD", C_SD, paste0(PM, "e6829f2afed2"), paste0(PM, "26778aefa564"),
  "SA", "SA", C_SA, paste0(PM, "fdebb8564dd8"), paste0(PM, "2b1342db3549"),
  "intact", "intact", C_GRAY, paste0(CL, "b8126b3cee2e"), paste0(CL, "e700f75eae04"),
  "broad", "broad", C_BROAD, paste0(CL, "5d3c56234fb4"), paste0(CL, "7e163cc4c001"),
  "hub", "learned", C_LEARNED, paste0(CS, "b91bfe93a18d"), paste0(CS, "c9cc8054983b")) %>%
  rowwise() %>% mutate(two = ledger(two_id)$est, fin = ledger(fin_id)$est, n = ledger(two_id)$n,
                       filled = confirmatory(two_id) || confirmatory(fin_id)) %>% ungroup() %>%
  mutate(y = rev(seq_len(n())), fill = ifelse(filled, col, "white"))
stopifnot(rescue$n == c("8", "6", "12", "70", "13"), all(ledger_field(rescue$two_id[1], 7) == "8"))
cues <- fd("items_fig4d_patient_cueing.csv")
stopifnot(sum(cues$group == "SD") == 8, sum(cues$group == "SA") == 6,
          all(cues$rescued_two >= 0 & cues$rescued_final <= 1))
cues <- cues %>% group_by(group) %>% arrange(rescued_two, .by_group = TRUE) %>%
  mutate(y = rescue$y[match(group, rescue$row)] + seq(-0.3, 0.3, length.out = n()),
         col = ifelse(group == "SD", C_SD, C_SA)) %>% ungroup()
ends <- rescue %>% pivot_longer(c(two, fin), names_to = "cue", values_to = "x")
sd <- rescue[rescue$row == "SD", ]   # kept for the checks below
pC <- ggplot() +
  geom_segment(data = cues, aes(x = rescued_two, xend = rescued_final, y = y, yend = y, colour = I(col)),
               linewidth = 0.3, alpha = 0.5, lineend = "round") +
  geom_segment(data = rescue, aes(x = two, xend = fin, y = y, yend = y, colour = I(col)), linewidth = 1.0, lineend = "butt") +
  geom_pentagon(data = filter(ends, kind %in% c("SD", "SA")), aes(x, y, colour = I(col), fill = I(fill)), size = 2.7, linewidth = 0.4) +
  geom_point(data = filter(ends, kind == "intact"), aes(x, y, colour = I(col), fill = I(fill)), shape = 21, size = 1.7, stroke = 0.7) +
  geom_point(data = filter(ends, kind %in% c("broad", "learned")), aes(x, y, colour = I(col), fill = I(fill)), shape = 23, size = 2.3, stroke = 0.7) +
  geom_text(data = rescue, aes(x = 1.21, y = y, label = n, colour = I(col)), hjust = 1, size = pt_size(8), family = "Arial") +
  annotate("text", x = 1.21, y = 5.43, label = "n", hjust = 1, size = pt_size(8), fontface = "italic", colour = "#555555",
           family = "Arial") +
  lab(0.01, 5.68, "two letters", col = "#555555", hjust = 0) +     # the two cues name the two ends of every bar
  lab(1.21, 5.68, "all but last", col = "#555555", hjust = 1) +
  scale_x_continuous("Share rescued", limits = c(0, 1.22), breaks = c(0, 0.5, 1), labels = c("0", ".5", "1"),
                     expand = expansion(0)) +
  scale_y_continuous(NULL, breaks = rescue$y, labels = rescue$row, limits = c(0.55, 5.95), expand = expansion(0)) +
  labs(title = "Cues still rescue", tag = "C") + base +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(), axis.text.y = element_text(size = 8.5))

# ---- assemble and export: A across the top, B (five facets) and C below
W_MM <- 180; H_MM <- 128
# One white page behind the whole figure and no background rectangle per panel: a stroked or filled rectangle
# under each panel would be read by the collision audit as a path crossing the panel letters.
fig <- (pA + pB + pC + plot_layout(design = "AA\nBC", widths = c(3.7, 1), heights = c(1, 1.3)) +
          plot_annotation(theme = theme(plot.background = element_rect(fill = "white", colour = NA)))) *
  theme(plot.background = element_blank())   # `*` reaches the three panels only, `&` would blank the page too
export_figure(fig, "Figure_5", W_MM, H_MM)
# cairo keeps one text object across consecutive labels; split it so the collision audit sees one box per label
pdf_path <- file.path(ROOT, "manuscript/figures/Figure_5.pdf")
split_out <- system2("python", c(shQuote(file.path(dirname(.kit_file), "split_pdf_text.py")), shQuote(pdf_path), "--check"),
                     stdout = TRUE, stderr = TRUE)
cat(split_out, sep = "\n")
if (!is.null(attr(split_out, "status")) && attr(split_out, "status") != 0) stop("split_pdf_text failed")

# ---- QA: alignment manifest and gate (1.5 pt), word count
QA <- file.path(ROOT, "docs/paper-prep/figures/qa")
boxes <- panel_boxes_pt(fig, W_MM, H_MM, c(
  a = "^panel-1\\.", b = "^panel; panel-1-1", b1 = "^panel-1-1\\.", b2 = "^panel-2-1\\.", b3 = "^panel-3-1\\.",
  b4 = "^panel-4-1\\.", b5 = "^panel-5-1\\.", c = "^panel-3\\."))
grid_pos <- list(a = list(grid = "patchwork-grid-1", r = c(0L, 1L), c = c(0L, 2L)),
                 b = list(grid = "patchwork-grid-1", r = c(1L, 2L), c = c(0L, 1L)),
                 c = list(grid = "patchwork-grid-1", r = c(1L, 2L), c = c(1L, 2L)))
for (i in 1:5) grid_pos[[paste0("b", i)]] <- list(grid = "facet-grid-b", r = c(0L, 1L), c = c(i - 1L, i))
manifest <- file.path(QA, "Figure_5.alignment-layout.json")
write_alignment_manifest(boxes, grid_pos, W_MM, H_MM, manifest, exemptions = list())
audit <- "C:/Users/wanghe/.claude/skills/nature-figure/scripts/audit_panel_alignment.py"
out <- system2("python", c(shQuote(audit), shQuote(manifest), "--json-out", shQuote(file.path(QA, "Figure_5.alignment.json")),
                           "--overlay-svg", shQuote(file.path(QA, "Figure_5.alignment.svg")),
                           "--tolerance-pt", "1.5", "--gutter-tolerance-pt", "1.5", "--strict"), stdout = TRUE, stderr = TRUE)
cat(out, sep = "\n")
if (!is.null(attr(out, "status")) && attr(out, "status") != 0) stop("alignment gate failed")
wc <- panel_word_count(file.path(ROOT, "manuscript/figures/Figure_5.svg"))
cat("words inside panels:", wc$n, "\n"); cat(wc$words, "\n")
stopifnot(wc$n <= 70)
