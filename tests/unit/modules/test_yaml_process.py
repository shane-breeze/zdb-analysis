import pytest
import mock

from zdb.modules.yaml_process import yaml_read, create_query_string

@pytest.fixture()
def path():
    return "dummy.yaml"

@pytest.fixture()
def data_string():
    return (
        """query:\n"""
        """    index: ["idx1", "idx2"]\n"""
        """    template: "SELECT {columns} FROM Events WHERE {selection} GROUP BY {groupby}"\n"""
        """    aliases:\n"""
        """        alias1: "unalias1"\n"""
        """        alias2: "unalias2"\n"""
        """    histograms:\n"""
        """        hist1:\n"""
        """            columns:\n"""
        """                - "idx1, idx2"\n"""
        """                - "item1 as it1"\n"""
        """                - "item2 as it2"\n"""
        """                - "{weight} as w"\n"""
        """            selection:\n"""
        """                sele1:\n"""
        """                    selection: "cut1==1"\n"""
        """                    weights: "weight1*weight2"\n"""
        """                sele2:\n"""
        """                    selection: "cut2==1"\n"""
        """                    weights: "weight2*weight3*{alias1}"\n"""
        """            groupby:\n"""
        """                "idx1, idx2"\n"""
        """database:\n"""
        """    - "db1.db"\n"""
        """    - "db2.db"\n"""
    )

@pytest.fixture()
def data_dict():
    return {
        "query": {
            "index": ["idx1", "idx2"],
            "template": "SELECT {columns} FROM Events WHERE {selection} GROUP BY {groupby}",
            "aliases": {
                "alias1": "unalias1",
                "alias2": "unalias2",
            },
            "histograms": {
                "hist1": {
                    "columns": [
                        "idx1, idx2",
                        "item1 as it1",
                        "item2 as it2",
                        "{weight} as w",
                    ],
                    "selection": {
                        "sele1": {
                            "selection": "cut1==1",
                            "weights": "weight1*weight2",
                        },
                        "sele2": {
                            "selection": "cut2==1",
                            "weights": "weight2*weight3*{alias1}",
                        },
                    },
                    "groupby": "idx1, idx2",
                },
            },
        },
        "database": [
            "db1.db",
            "db2.db",
        ],
    }

def test_yaml_read(data_string, data_dict, path):
    mocked_open = mock.mock_open(read_data=data_string)
    with mock.patch("__builtin__.open", mocked_open):
        result = yaml_read(path)
    assert result == data_dict

@pytest.mark.parametrize(
    "hist,queries", ([
        "hist1", [
            "SELECT idx1, idx2, item1 as it1, item2 as it2, weight1*weight2 as w FROM Events WHERE cut1==1 GROUP BY idx1, idx2",
            "SELECT idx1, idx2, item1 as it1, item2 as it2, weight2*weight3*unalias1 as w FROM Events WHERE cut2==1 GROUP BY idx1, idx2",
        ]
    ],)
)
def test_create_query_string(data_dict, hist, queries):
    query = data_dict["query"]
    hist_dict = query["histograms"][hist]
    result = create_query_string(
        query["template"], hist_dict, aliases=query["aliases"],
    )
    assert sorted(queries) == sorted(result)
