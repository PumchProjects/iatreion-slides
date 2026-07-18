#import "@preview/touying:0.7.4": *
#import themes.dewdrop: *

#import "@preview/numbly:0.1.0": numbly
#import "@preview/pinit:0.2.2": *

#let config = toml("/config.toml")

#let primary = rgb(95, 33, 103)

#let serif = ("Libertinus Serif", "Source Han Serif")
#let sans-serif = ("Arial", "Source Han Sans")
// Don't use Helvetica otherwise the dots in the mini-slide headings do not align

#let eg = [_e.g.,_]
#let ie = [_i.e.,_]
#let etal = [_et al._]

#let newpage() = [
  == <touying:hidden>
]

// #empty-slide[]

#let purple-it(body) = {
  set text(fill: primary)
  body
}

#let gray-it(body) = {
  set text(fill: gray)
  body
}

#let greek(body) = {
  set text(font: "SBL Greek")
  body
}

#let syriac(body) = {
  set text(font: "Serto Antoch Bible")
  body
}

#let thanks(body) = {
  focus-slide({
    set text(font: serif)
    block(below: 1em, text(2em)[Thanks])
    body
  })
}
