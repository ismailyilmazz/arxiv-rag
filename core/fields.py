ALIASES = {
    "alg-geom": "math", "q-alg": "math", "dg-ga": "math", "funct-an": "math",
    "chao-dyn": "nlin", "solv-int": "nlin", "patt-sol": "nlin", "adap-org": "nlin", "comp-gas": "nlin",
    "cmp-lg": "cs", "mtrl-th": "cond-mat", "supr-con": "cond-mat",
    "chem-ph": "physics", "atom-ph": "physics", "acc-phys": "physics", "plasm-ph": "physics", "ao-sci": "physics",
    "bayes-an": "stat",
    "hep-ph": "hep", "hep-th": "hep", "hep-ex": "hep", "hep-lat": "hep",
    "nucl-th": "nucl", "nucl-ex": "nucl",
}


def field_of(category: str) -> str:
    archive = (category or "").split(".")[0]
    return ALIASES.get(archive, archive)
