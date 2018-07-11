from load_tests import (
    nose2,
    data,
    build_graph,
    raises,
)


#####################################################
# Check parameters
#####################################################


@raises(TypeError)
def test_unknown_parameter():
    build_graph(data, hello='world')


@raises(ValueError)
def test_invalid_graphtype():
    build_graph(data, graphtype='hello world')
