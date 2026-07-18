#import "globals.typ": *

#show: dewdrop-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.footer,
  // navigation: "mini-slides",
  navigation: none,
  mini-slides: (height: 3em, linebreaks: false, x: 2em),
  primary: primary,
  subslide-preamble: self => block(
    text(
      1.2em,
      font: serif,
      weight: "bold",
      fill: self.colors.primary,
      utils.display-current-heading(depth: self.slide-level),
    ),
  ),
  config-info(
    title: text(font: serif, weight: "bold", 1em)[#config.report.title],
    subtitle: text(font: serif, weight: "bold", 1.2em, fill: rgb("#555"))[#config.report.subtitle],
    author: [#config.report.author],
    date: datetime.today(),
    // date: datetime(year: 1453, month: 5, day: 29),
    // institution: [],
    footer: text(font: serif)[#config.report.footer],
  ),
  config-common(
    datetime-format: "[year].[month].[day]",
    new-section-slide-fn: section => {
      touying-slide-wrapper(self => {
        touying-slide(
          self: self,
          {
            set align(center + horizon)
            set text(size: 2em, fill: self.colors.primary, font: serif, weight: "bold")
            set par(justify: false)
            utils.display-current-heading(level: 1)
          },
        )
      })
    },
    // new-section-slide-fn: none,
    // handout: true,
  ),
)

#set text(font: sans-serif, weight: "regular", 20pt, lang: "zh")

#set heading(numbering: numbly("{1}.", "{1}.{2}", "({3})", default: none))

#set par(justify: true)

#set underline(offset: .15em, stroke: .05em, evade: false)
#show regex("\{[\w/]+\}"): it => underline(it.text.slice(1, -1))

#set table(
  fill: (x, y) => if y == 0 {
    silver
  },
  align: horizon + center,
  inset: (x: 1.2em, y: 0.6em),
)

#show table.cell: it => {
  if it.y == 0 {
    strong(it)
  } else {
    it
  }
}

#show figure.where(kind: table): set figure.caption(position: top)

#set quote(block: true)

#let list-counter = counter("list")

#show list: it => {
  list-counter.step()

  context {
    set text(0.85em) if list-counter.get().first() >= 2
    it
  }
  list-counter.update(i => i - 1)
}

#show sym.circle.filled: it => box(move(dy: 1.2pt, it))
// Lower sym.circle.filled a bit to match sym.circle.small

#title-slide(extra: none)

// #outline-slide(title: text(font: serif, 1.2em)[目录], depth: 2)

#columns(1, outline(depth: 2, title: none))

#include "content.typ"

#thanks[Q & A]
