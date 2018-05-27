from . import (
    nose2,
    data,
    build_graph,
    raises,
)


#####################################################
# Check parameters
#####################################################


@raises(ValueError)
def test_invalid_graphtype():
    build_graph(data, graphtype='hello world')


if __name__ == "__main__":
    exit(nose2.run())
