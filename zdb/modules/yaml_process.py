import yaml

def yaml_read(path):
    with open(path, 'r') as f:
        return yaml.load(f)

def create_query_string(template, hist_dict, aliases={}):
    queries = {}
    for label, selection in hist_dict["selection"].items():
        weight = selection["weights"]
        columns = ", ".join(hist_dict["columns"]).format(Weight=weight)
        queries[label] = template.format(
            columns = columns,
            selection = selection["selection"],
            groupby = hist_dict["groupby"],
        ).format(**aliases)
    return queries
