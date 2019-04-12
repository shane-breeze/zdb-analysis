import yaml

def yaml_read(path):
    with open(path, 'r') as f:
        return yaml.load(f)

def create_query_string(template, hist_dict, aliases={}):
    columns = ", ".join(hist_dict["columns"]).format(**aliases)
    groupby = hist_dict["groupby"]
    return {
        label: template.format(
            columns=columns, selection=selection, groupby=groupby,
        ) for label, selection in hist_dict["selection"].items()
    }
