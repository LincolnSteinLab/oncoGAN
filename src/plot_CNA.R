suppressMessages(library(argparse))
suppressMessages(library(dplyr))
suppressMessages(library(ggplot2))

# CLI options
parser <- ArgumentParser()
parser$add_argument("--cna",
                    help = "TSV file with CNA oputput")
parser$add_argument("--output",
                    help = "Path where save the image")
args <- parser$parse_args()

# Open CNA results
tb <- readr::read_delim(args$cna,
                        show_col_types = FALSE) %>% 
  dplyr::mutate(chrom = stringr::str_remove(chrom, "chr"),
                chrom = factor(chrom,
                               levels = c(1:22, "X", "Y")))

# Create chromosome positions
pad_tb <- tb %>% 
  dplyr::group_by(chrom) %>% 
  dplyr::summarise(chr_length = max(end),
                   .groups = "drop") %>% 
  dplyr::arrange(chrom) %>% 
  dplyr::mutate(chr_cumstart = cumsum(lag(chr_length, default = 0)),
                chr_cumend = lead(chr_cumstart),
                chr_cumend = ifelse(is.na(chr_cumend),
                                    chr_cumstart+chr_length,
                                    chr_cumend)) %>%
  dplyr::select(chrom, chr_cumstart, chr_cumend)

# Process CNAs
gg_tb <- tb %>% 
  dplyr::left_join(pad_tb %>% dplyr::select(chrom, chr_cumstart),
                   by = "chrom") %>% 
  dplyr::mutate(start = start + chr_cumstart,
                end = end + chr_cumstart,
                cn_a = cn_a + 0.15,
                cn_b = cn_b - 0.15) %>% 
  dplyr::select(-sample_id, -tumor, -chr_cumstart) %>% 
  tidyr::pivot_longer(cols = c(cn_a, cn_b),
                      names_to = "allele",
                      values_to = "cn") %>% 
  dplyr::mutate(allele = factor(allele,
                                levels = c("cn_a", "cn_b"),
                                labels = c("Allele A", "Allele B")))

# Extract variables
tumor <- tb$tumor[1]
sample_id <- tb$sample_id[1]
y_breaks <- pretty(c(0, max(gg_tb$cn)), bounds = FALSE)

# Plot 
gg_cnas <- gg_tb %>%
  ggplot() +
  ## CNAs
  geom_segment(aes(x = start, xend = end,
                   y = cn,
                   color = allele),
               linewidth = 0.8) +
  ## Chrom separation line
  geom_segment(data = pad_tb %>% dplyr::filter(chrom != "1"),
               aes(x = chr_cumstart,
                   y = -0.2, yend = max(y_breaks)),
               linetype = "dashed", linewidth = 0.2,
               color = "darkgrey") +
  ## Chrom names
  geom_text(data = pad_tb,
            aes(x = chr_cumstart+(chr_cumend-chr_cumstart)/2,
                y = rep(c(max(y_breaks), max(y_breaks)+0.4),
                        length.out = nrow(pad_tb)),
                label = chrom),
            size = 2,
            color = "#00000090") +
  ## X axis line
  geom_segment(x = 0, xend = max(gg_tb$end),
               y = -0.3,
               inherit.aes = FALSE, linewidth = 0.3) +
  ## Y axis line
  geom_segment(x = -10^7.8,
               y = 0, yend = max(y_breaks),
               inherit.aes = FALSE, linewidth = 0.3) +
  ## Coordinate options
  coord_cartesian(xlim = c(-10^7.8, max(gg_tb$end)),
                  ylim = c(-0.3, max(y_breaks))) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = expansion(add = c(0, 1)),
                     breaks = y_breaks) +
  ## Other aesthetic options
  scale_color_manual(values = c("#4f9b8f", "#e8a66c")) +
  labs(title = paste(tumor, sample_id, sep = " - "),
       x = "Genome",
       y = "CNA",
       color = "") +
  ## Theme options
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 11),
        panel.grid = element_blank(),
        axis.title = element_text(size = 9),
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 7),
        axis.ticks.y = element_line(linewidth = 0.3),
        legend.text = element_text(size = 6),
        legend.position = "top")

ggsave(args$output,
       plot = gg_cnas,
       width = 7, height = 3.5, dpi = 1000)
