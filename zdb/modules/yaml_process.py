import yaml

def yaml_read(path):
    with open(path, 'r') as f:
        return yaml.full_load(f)

def create_query_string(template, hist_dict, aliases={}):
    return [
        template.format(
            columns = ", ".join(hist_dict["columns"]).format(
                weight = selection["weights"],
                selection_name = label,
            ),
            table = hist_dict["table"],
            selection = selection["selection"],
            groupby = hist_dict["groupby"],
        ).format(**aliases)
        for label, selection in hist_dict["selection"].items()
    ]
