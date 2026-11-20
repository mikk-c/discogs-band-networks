set terminal pdfcairo enhanced size 3,2.5 crop color solid rounded linewidth 1

unset key
set view map scale 1
set style data lines
set xtics border in scale 0,0 mirror rotate 90  autojustify
set ytics border in scale 0,0 mirror norotate  autojustify
set ztics border in scale 0,0 nomirror norotate  autojustify

set xrange [-0.5:5.5]
set yrange [5.5:-0.5]
set cbrange [0:1]

set xtics ("NC Cl" 0, "NC T6" 1, "DF T6" 2, "Na T6" 3, "DF Cl" 4, "Na Cl" 5)
set ytics ("NC Cl" 0, "NC T6" 1, "DF T6" 2, "Na T6" 3, "DF Cl" 4, "Na Cl" 5)
set cblabel "Jaccard"

set palette defined (0 "#fff5f0", 0.25 "#fcbba1", 0.5 "#fb6a4a", 0.75 "#cb181d", 1 "#67000d")

set tmargin 0
set bmargin 1

set output "fig_8.pdf"
splot "fig_8.tsv" matrix with image
