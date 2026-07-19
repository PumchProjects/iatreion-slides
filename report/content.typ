#import "globals.typ": *

#let read-tsv(path) = read(path).split("\n").map(it => it.split("\t"))
#let unit-root(unit) = "/" + unit.root
#let model-id(model) = model.name + model.at("suffix", default: "")

#let show-result(exp-root, log-file, roc-file) = {
  figure(
    grid(
      columns: 2,
      gutter: 2em,
      raw(read(exp-root + "/" + log-file).trim(), block: true, lang: "log"),
      image(exp-root + "/" + roc-file, height: 80%),
    ),
  )
}

#let show-weights(unit, setting, name, group) = {
  let final-root = (unit-root(unit), setting, "final", group).join("/")
  let (model-names, modalities, model-bests, model-weights) = ((), (), (:), ())
  for model in unit.models {
    let id = model-id(model)
    model-names.push(config.model_map.at(id))
    let tsv-path = (final-root, id, "eval", name, "weights.tsv").join("/")
    let tsv-data = read-tsv(tsv-path)
    let tsv-weights = tsv-data.map(it => float(it.at(1)))
    let max-index = tsv-weights.position(x => x == calc.max(..tsv-weights))
    modalities = tsv-data.map(it => config.modality_map.at(it.at(0)))
    model-bests.insert(id, tsv-data.at(max-index).at(0))
    model-weights.push(tsv-data.enumerate().map(it => {
      let (index, (_, weight)) = it
      if index == max-index { strong(weight) } else {weight}
    }))
  }
  let data = modalities.zip(..model-weights)
  (model-bests, slide[
    === 各模态权重
    // #set text(0.9em)
    #set par(justify: false)
    #figure(
      table(
        columns: unit.models.len() + 1,
        table.header("模态", ..model-names),
        ..data.flatten(),
      ),
    )
  ])
}

#let show-importance(unit, setting, name, group, model-bests) = {
  let training-root = (unit-root(unit), setting, "training", unit.train_name, group).join("/")
  let models = unit.models.filter(model => config.collect.importance_models.contains(model.name))
  let model-names = models.map(model => config.model_map.at(model-id(model)))
  for ref in unit.refs {
    let model-scores = models.map(model => {
      let id = model-id(model)
      let modality = model-bests.at(id)
      let score-file = "scores_" + modality + ".tsv"
      let score-path = (training-root, id, ref, score-file).join("/")
      read-tsv(score-path)
    })
    slide[
      === 重要变量（从左至右分别为 #model-names.join(", ")）
      #set text(0.7em)
      #set par(justify: false)
      #figure(
        grid(
          columns: model-names.len(),
          gutter: 1em,
          ..model-scores.map(scores => figure(
            table(
              columns: 2,
              table.header([变量], [重要性]),
              ..scores.flatten(),
            ),
          ))
        ),
      )
    ]
  }
}

#let show-rules(unit, setting, name, group) = {
  let final-root = (unit-root(unit), setting, "final", group).join("/")
  for model in unit.models.filter(model => model.name == "rrl") {
    let id = model-id(model)
    let rule-path = (final-root, id, "eval", name, "rules.tsv").join("/")
    let rule-data = read-tsv(rule-path)
    slide[
      === 重要规则（#config.model_map.at(id)）
      #set text(0.57em)
      #set par(justify: false)
      #show regex("\band\b"): text.with(weight: "bold", orange)
      #show regex("\bnot\b"): text.with(weight: "bold", green)
      #show regex("\bor\b"): text.with(weight: "bold", fuchsia)
      #figure(
        table(
          columns: 4,
          table.header([模态], [标签], [分数], [规则]),
          ..rule-data.slice(1, count: 5).map(it => {
            let (modality, label, score, rule) = it
            (
              config.modality_map.at(modality),
              config.group_map.at(label),
              score,
              rule,
            )
          }).flatten(),
        ),
      )
    ]
  }
}

#let training-results(unit, setting, name, group) = {
  let training-root = (unit-root(unit), setting, "training", unit.train_name, group).join("/")
  for ref in unit.refs {
    for model in unit.models {
      let id = model-id(model)
      let model-name = [#config.model_map.at(id)（#config.ref_map.at(ref)）]
      let exp-root = (training-root, "result-replay", id, ref, name).join("/")
      let ref-name = "all_" + ref.replace("-", "_")
      let log-file = "train_ci_" + ref-name + "_youden.log"
      let roc-file = "roc_" + ref-name + "_original.png"
      slide[
        === #model-name
        #show-result(exp-root, log-file, roc-file)
      ]
    }
  }
}

#let final-results(unit, setting, name, group) = {
  let final-root = (unit-root(unit), setting, "final", group).join("/")
  for model in unit.models {
    let id = model-id(model)
    let model-name = [#config.model_map.at(id)（外部验证）]
    let exp-root = (final-root, id, "eval", name).join("/")
    let log-file = "results.log"
    let roc-file = "roc_" + name + ".png"
    slide[
      === #model-name
      #show-result(exp-root, log-file, roc-file)
    ]
  }
}

#let results(unit) = {
  for setting in unit.settings {
    for name in unit.names {
      let setting-name = (
        config.name_map.at(name),
        config.setting_map.at(setting),
      ).join(", ")
      [= #setting-name]
      for group in unit.groups {
        let group-name = group.split("_").map(it => config.group_map.at(it)).join(" : ")
        [== #group-name]
        training-results(unit, setting, name, group)
        final-results(unit, setting, name, group)
        let (model-bests, weights) = show-weights(unit, setting, name, group)
        weights
        show-importance(unit, setting, name, group, model-bests)
        show-rules(unit, setting, name, group)
      }
    }
  }
}

#for unit in config.units {
  results(unit)
}
